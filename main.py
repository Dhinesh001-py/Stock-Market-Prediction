import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import plotly.graph_objs as go
from dash import Dash, dcc, html, dash_table

# Step 1: Data collection and preprocessing
stock_data = yf.download('AAPL', start='2023-01-01', end='2025-01-01', auto_adjust=True)
stock_data.reset_index(inplace=True)
stock_data.ffill(inplace=True)  # <- fixed future warning

stock_data['MA20'] = stock_data['Close'].rolling(window=20).mean()
stock_data['MA50'] = stock_data['Close'].rolling(window=50).mean()
stock_data['Target'] = stock_data['Close'].shift(-1)
stock_data.dropna(inplace=True)

# Step 2: Feature selection and model training
features = ['Close', 'MA20', 'MA50']
X = stock_data[features]
y = stock_data['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")

# Step 3: Prepare data for visualization
results_df = pd.DataFrame({
    'Date': stock_data['Date'].iloc[-len(y_test):].dt.strftime('%Y-%m-%d'),
    'Actual Price': y_test.values,
    'Predicted Price': predictions
})

# Step 4: Create Plotly graph
graph = go.Figure()
graph.add_trace(go.Scatter(x=results_df['Date'], y=results_df['Actual Price'],
                           mode='lines', name='Actual Price'))
graph.add_trace(go.Scatter(x=results_df['Date'], y=results_df['Predicted Price'],
                           mode='lines', name='Predicted Price'))
graph.update_layout(title='AAPL Stock Price Prediction',
                    xaxis_title='Date',
                    yaxis_title='Stock Price (USD)')

# Step 5: Create Dash app
app = Dash(__name__)

app.layout = html.Div(children=[
    html.H1("AAPL Stock Price Prediction Dashboard"),
    dcc.Graph(figure=graph),
    html.H2("Actual vs Predicted Table"),
    dash_table.DataTable(
        data=results_df.to_dict('records'),
        columns=[{"name": i, "id": i} for i in results_df.columns],
        style_table={'overflowX': 'auto'},
        style_cell={'padding': '8px', 'textAlign': 'center'},
        style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'}
    )
])

# Step 6: Run the app
if __name__ == '__main__':
    app.run(debug=True)  # <- updated line
