let riskLevel = 30;
const riskElement = document.getElementById("risk-level");
const riskBar = document.getElementById("risk-bar");
const transactionsLog = document.getElementById("transactions-log");

function performAction(action) {
    let change = 0;
    let logMessage = "";
    
    switch (action) {
        case 'deposit':
            change = 5;
            logMessage = "Deposited funds";
            break;
        case 'buy':
            change = -10;
            logMessage = "Bought assets";
            break;
        case 'sell':
            change = -5;
            logMessage = "Sold assets";
            break;
        case 'withdraw':
            change = 15;
            logMessage = "Withdrew funds";
            break;
    }
    
    updateRisk(change);
    logTransaction(logMessage);
}

function updateRisk(change) {
    riskLevel = Math.max(0, Math.min(100, riskLevel + change));
    riskElement.textContent = `${riskLevel}%`;
    
    riskBar.styles.width = `${riskLevel}%`;
    riskBar.styles.backgroundColor = getRiskColor(riskLevel);
}

function getRiskColor(level) {
    if (level < 40) return "green";
    if (level < 70) return "orange";
    return "red";
}

function logTransaction(message) {
    const entry = document.createElement("p");
    entry.textContent = message;
    transactionsLog.prepend(entry);
}
