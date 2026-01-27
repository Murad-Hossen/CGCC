/**
 * Leaderboard Configuration
 * Modify these settings to customize the leaderboard behavior
 */
const LEADERBOARD_CONFIG = {
    // JSON file path
    jsonPath: 'leaderboard.json',
    
    // Field to use for sorting (descending order)
    sortBy: 'validation_f1_score',
    
    // Field to use as the primary score (will be highlighted)
    primaryScoreField: 'validation_f1_score',
    
    // Fields to exclude from display (will always show team_name and rank)
    excludeFields: [],
    
    // Field name mappings (key: JSON field name, value: display name)
    fieldNames: {
        'team_name': 'Team',
        'validation_accuracy': 'Validation Accuracy',
        'validation_f1_score': 'Validation F1 Score',
        'timestamp': 'Submission Time'
    },
    
    // Field formatters (key: JSON field name, value: formatting function)
    fieldFormatters: {
        'timestamp': (value) => {
            // Format timestamp from "YYYY-MM-DD HH:MM:SS" to readable format
            try {
                const date = new Date(value.replace(' ', 'T'));
                return date.toLocaleString('en-US', {
                    month: 'long',
                    day: 'numeric',
                    year: 'numeric',
                    hour: '2-digit',
                    minute: '2-digit',
                    second: '2-digit',
                    hour12: false
                }).replace(',', ' at');
            } catch (e) {
                return value;
            }
        },
        'validation_accuracy': (value) => {
            return typeof value === 'number' ? value.toFixed(6) : value;
        },
        'validation_f1_score': (value) => {
            return typeof value === 'number' ? value.toFixed(6) : value;
        }
    }
};

/**
 * Leaderboard Manager
 * Handles loading and rendering leaderboard data
 */
class LeaderboardManager {
    constructor(config) {
        this.config = config;
        this.data = [];
    }
    
    /**
     * Load leaderboard data from JSON file
     */
    async loadData() {
        try {
            const response = await fetch(this.config.jsonPath);
            if (!response.ok) {
                throw new Error(`Failed to load leaderboard data: ${response.statusText}`);
            }
            this.data = await response.json();
            return this.data;
        } catch (error) {
            console.error('Error loading leaderboard data:', error);
            throw error;
        }
    }
    
    /**
     * Sort data by the configured sort field
     */
    sortData() {
        if (!this.config.sortBy || !this.data.length) {
            return;
        }
        
        this.data.sort((a, b) => {
            const aVal = a[this.config.sortBy];
            const bVal = b[this.config.sortBy];
            
            // Handle null/undefined values
            if (aVal == null && bVal == null) return 0;
            if (aVal == null) return 1;
            if (bVal == null) return -1;
            
            // Sort in descending order
            if (typeof aVal === 'number' && typeof bVal === 'number') {
                return bVal - aVal;
            }
            
            return String(bVal).localeCompare(String(aVal));
        });
    }
    
    /**
     * Get all column names from the data (excluding excluded fields)
     */
    getColumns() {
        if (!this.data.length) {
            return [];
        }
        
        const columns = [];
        const firstItem = this.data[0];
        
        // Always include team_name first
        if ('team_name' in firstItem) {
            columns.push('team_name');
        }
        
        // Add other fields
        for (const key in firstItem) {
            if (key !== 'team_name' && 
                !this.config.excludeFields.includes(key)) {
                columns.push(key);
            }
        }
        
        return columns;
    }
    
    /**
     * Format a field value using configured formatters
     */
    formatValue(fieldName, value) {
        if (this.config.fieldFormatters[fieldName]) {
            return this.config.fieldFormatters[fieldName](value);
        }
        
        // Default formatting
        if (typeof value === 'number') {
            return value.toFixed(6);
        }
        
        return value;
    }
    
    /**
     * Get display name for a field
     */
    getDisplayName(fieldName) {
        return this.config.fieldNames[fieldName] || 
               fieldName.split('_').map(word => 
                   word.charAt(0).toUpperCase() + word.slice(1)
               ).join(' ');
    }
    
    /**
     * Get CSS class for a field
     */
    getFieldClass(fieldName) {
        const classes = [];
        
        if (fieldName === this.config.primaryScoreField) {
            classes.push('score', 'primary-score');
        } else if (typeof this.data[0]?.[fieldName] === 'number') {
            classes.push('score');
        }
        
        if (fieldName === 'team_name') {
            classes.push('team-name');
        }
        
        return classes.join(' ');
    }
    
    /**
     * Get medal emoji for rank
     */
    getMedal(rank) {
        if (rank === 1) return '🥇';
        if (rank === 2) return '🥈';
        if (rank === 3) return '🥉';
        return '';
    }
    
    /**
     * Get rank CSS class
     */
    getRankClass(rank) {
        if (rank <= 3) {
            return `rank-${rank}`;
        }
        return '';
    }
    
    /**
     * Render the leaderboard table
     */
    render() {
        const tableHeader = document.getElementById('table-header');
        const tableBody = document.getElementById('table-body');
        const lastUpdated = document.getElementById('last-updated');
        
        if (!tableHeader || !tableBody) {
            console.error('Table elements not found');
            return;
        }
        
        // Clear existing content
        tableHeader.innerHTML = '';
        tableBody.innerHTML = '';
        
        if (!this.data || this.data.length === 0) {
            tableBody.innerHTML = '<tr><td colspan="100%" class="empty">No leaderboard data available</td></tr>';
            if (lastUpdated) {
                lastUpdated.textContent = 'No data available';
            }
            return;
        }
        
        // Sort data
        this.sortData();
        
        // Get columns
        const columns = this.getColumns();
        
        // Render header
        const headerRow = document.createElement('tr');
        headerRow.innerHTML = '<th class="rank">Rank</th>';
        
        columns.forEach(column => {
            const th = document.createElement('th');
            const displayName = this.getDisplayName(column);
            const fieldClass = this.getFieldClass(column);
            
            th.textContent = displayName;
            if (fieldClass) {
                th.className = fieldClass;
            }
            
            headerRow.appendChild(th);
        });
        
        tableHeader.appendChild(headerRow);
        
        // Render rows
        this.data.forEach((entry, index) => {
            const rank = index + 1;
            const row = document.createElement('tr');
            row.style.animationDelay = `${(index + 1) * 0.1}s`;
            
            // Rank cell
            const rankCell = document.createElement('td');
            rankCell.className = `rank ${this.getRankClass(rank)}`;
            const medal = this.getMedal(rank);
            rankCell.innerHTML = `${medal ? `<span class="medal">${medal} </span>` : ''}${rank}`;
            row.appendChild(rankCell);
            
            // Data cells
            columns.forEach(column => {
                const cell = document.createElement('td');
                const value = entry[column];
                const fieldClass = this.getFieldClass(column);
                
                if (fieldClass) {
                    cell.className = fieldClass;
                }
                
                cell.textContent = value != null ? this.formatValue(column, value) : '-';
                row.appendChild(cell);
            });
            
            tableBody.appendChild(row);
        });
        
        // Update last updated timestamp
        if (lastUpdated) {
            const now = new Date();
            const formattedDate = now.toLocaleString('en-US', {
                month: 'long',
                day: 'numeric',
                year: 'numeric',
                hour: '2-digit',
                minute: '2-digit',
                second: '2-digit',
                hour12: false
            }).replace(',', ' at');
            lastUpdated.textContent = `Last updated: ${formattedDate}`;
        }
    }
    
    /**
     * Initialize and load the leaderboard
     */
    async init() {
        try {
            await this.loadData();
            this.render();
        } catch (error) {
            const tableBody = document.getElementById('table-body');
            if (tableBody) {
                tableBody.innerHTML = `<tr><td colspan="100%" class="empty">Error loading leaderboard: ${error.message}</td></tr>`;
            }
        }
    }
}

// Initialize leaderboard when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    const leaderboard = new LeaderboardManager(LEADERBOARD_CONFIG);
    leaderboard.init();
});

