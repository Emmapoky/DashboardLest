from flask import Flask, render_template
import plotly.graph_objects as go
import json

app = Flask(__name__)

@app.route('/')
@app.route('/dashboard')
def dashboard():
    # Sample data for the time history graph
    time_data = {
        'x': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
        'y1': [100, 120, 140, 145, 142, 138],
        'y2': [90, 95, 110, 115, 112, 108]
    }
    
    # Create the plotly figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_data['x'], y=time_data['y1'], name='Series 1'))
    fig.add_trace(go.Scatter(x=time_data['x'], y=time_data['y2'], name='Series 2'))
    
    # Update layout for dark theme
    fig.update_layout(
        paper_bgcolor='rgba(26, 32, 44, 0)',
        plot_bgcolor='rgba(26, 32, 44, 0)',
        font_color='#fff',
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False
    )
    
    # Sample transaction data
    transactions = [
        {'user_id': '69231731', 'date': '04/06/2025', 'stock': 'S&P500', 'pl': '+1 USD', 'withdraw': '', 'deposit': '+501 USD'},
        {'user_id': '69231731', 'date': '04/06/2025', 'stock': 'S&P500', 'pl': '-2 USD', 'withdraw': '-498 USD', 'deposit': ''},
    ]
    
    return render_template('dashboard.html', plot_json=json.dumps(fig.to_dict()), transactions=transactions, risk_score=80)


@app.route('/inbox')
def inbox():
    manual_interventions = [
        {'user_id': '69231731', 'risk': 80, 'problem': 'Similar IP Addresses'},
        {'user_id': '69231731', 'risk': 60, 'problem': 'VPN IP Detected'},
    ]

    ai_scanned_users = [
        {'user_id': '69231731', 'risk': 70, 'problem': 'Repeated Odd-Hour Transactions'},
        {'user_id': '69231731', 'risk': 65, 'problem': 'Similar Activity with # 69231731'},
    ]

    return render_template('inbox.html', manual_interventions=manual_interventions, ai_scanned_users=ai_scanned_users)


if __name__ == '__main__':
    app.run(debug=True)
