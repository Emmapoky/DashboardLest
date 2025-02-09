# App.py

from flask import Flask, render_template
# import plotly.graph_objects as go
import json

app = Flask(__name__)

# transactions = []
# risk_percentage = 30
# transaction_order = ['deposit', 'buy', 'sell', 'withdraw']
# current_step = 0

# @app.route('/')
# def frauddemo():
#     return render_template('fraud-demo.html')

# # Sample data for inbox notifications
# INBOX_NOTIFICATIONS = 4

# # Sample data for inbox tables
# manual_interventions = [
#     {'user_id': '69231731', 'risk': 80, 'problem': 'Similar IP Addresses'},
#     {'user_id': '69231731', 'risk': 60, 'problem': 'VPN IP Detected'},
#     {'user_id': '69231731', 'risk': 50, 'problem': 'Similar IP Addresses'},
#     {'user_id': '69231731', 'risk': 70, 'problem': 'Repeated Odd-Hour Transactions'},
# ]

# ai_scanned_users = [
#     {'user_id': '69231731', 'risk': 80, 'problem': 'Similar IP Addresses'},
#     {'user_id': '69231731', 'risk': 60, 'problem': 'VPN IP Detected'},
#     {'user_id': '69231731', 'risk': 50, 'problem': 'Similar IP Addresses'},
#     {'user_id': '69231731', 'risk': 70, 'problem': 'Repeated Odd-Hour Transactions'},
#     {'user_id': '69231731', 'risk': 65, 'problem': 'Similar Activity with # 69231731'},
#     {'user_id': '69231731', 'risk': 78, 'problem': 'Similar Activity with # 69231731'},
# ]

# @app.route('/dashboard')
# def dashboard():
#     # Sample data for the time history graph
#     time_data = {
#         'x': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
#         'y1': [100, 120, 140, 145, 142, 138],
#         'y2': [90, 95, 110, 115, 112, 108]
#     }
    
#     # Create the plotly figure
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=time_data['x'], y=time_data['y1'], name='Series 1'))
#     fig.add_trace(go.Scatter(x=time_data['x'], y=time_data['y2'], name='Series 2'))
    
#     # Update layout for dark theme
#     fig.update_layout(
#         paper_bgcolor='rgba(26, 32, 44, 0)',
#         plot_bgcolor='rgba(26, 32, 44, 0)',
#         font_color='#fff',
#         margin=dict(l=20, r=20, t=20, b=20),
#         showlegend=False
#     )
    
#     # Sample transaction data
#     transactions = [
#         {'user_id': '69231731', 'date': '04/06/2025', 'stock': 'S&P500', 'pl': '+1 USD', 'withdraw': '', 'deposit': '+501 USD'},
#         {'user_id': '69231731', 'date': '04/06/2025', 'stock': 'S&P500', 'pl': '-2 USD', 'withdraw': '-498 USD', 'deposit': ''},
#         {'user_id': '69231731', 'date': '04/06/2025', 'stock': 'S&P500', 'pl': '+1 USD', 'withdraw': '', 'deposit': '+101 USD'},
#         {'user_id': '69231731', 'date': '04/06/2025', 'stock': 'S&P500', 'pl': '+3 USD', 'withdraw': '', 'deposit': '+103 USD'},
#         {'user_id': '69231731', 'date': '04/06/2025', 'stock': 'S&P500', 'pl': '-1 USD', 'withdraw': '-599 USD', 'deposit': ''},
#         {'user_id': '69231731', 'date': '04/06/2025', 'stock': 'S&P500', 'pl': '+2 USD', 'withdraw': '', 'deposit': '+602 USD'},
#     ]
    
#     # Location and payment data
#     location_data = {
#         'user_id': '69231731',
#         'ip_address': '192.158.1.38',
#         'region': 'America'
#     }
    
#     payment_data = {
#         'user_id': '69231731',
#         'institution': 'Bank',
#         'user_id': '69231731',
#         'institution': 'Crypto'
#     }
    
#     return render_template('dashboard.html',
#                          notifications=INBOX_NOTIFICATIONS,
#                          plot_json=json.dumps(fig.to_dict()),
#                          transactions=transactions,
#                          location_data=location_data,
#                          payment_data=payment_data,
#                          risk_score=80)

# @app.route('/inbox')
# def inbox():
#     return render_template('inbox.html',
#                         notifications=INBOX_NOTIFICATIONS,
#                         manual_interventions=manual_interventions,
#                         ai_scanned_users=ai_scanned_users,
#                         current_page="inbox")

# @app.route('/review-form/<user_id>')
# def review_form(user_id):
#     # Find the user data from either manual_interventions or ai_scanned_users
#     user_data = None
#     for user in manual_interventions + ai_scanned_users:
#         if user['user_id'] == user_id:
#             user_data = user
#             break
    
#     if user_data is None:
#         return redirect(url_for('inbox'))
    
#     return render_template('review-form.html',
#                          user_data=user_data,
#                          notifications=INBOX_NOTIFICATIONS)

# @app.route('/submit-review/<user_id>', methods=['POST'])
# def submit_review(user_id):
#     if request.method == 'POST':
#         # Get form data
#         risk_assessment = request.form.get('risk_assessment')
#         action_taken = request.form.get('action_taken')
#         notes = request.form.get('notes')
        
#         # Here you would typically save this data to a database
#         # For now, we'll just print it
#         print(f"Review submitted for user {user_id}:")
#         print(f"Risk Assessment: {risk_assessment}")
#         print(f"Action Taken: {action_taken}")
#         print(f"Notes: {notes}")
        
#         # Redirect back to inbox after submission
#         # return redirect(url_for('inbox'))
