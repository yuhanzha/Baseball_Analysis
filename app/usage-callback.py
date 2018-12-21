import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import dash_table_experiments as dt
import pandas as pd
import config

app = dash.Dash()


df1 = pd.read_csv('pitch_join_display.csv')
df2 = pd.read_csv('pitch_type.csv')
df3 = pd.read_csv('team_gen1.csv')
df4 = pd.read_csv('team_stats.csv')
df5 = pd.read_csv('award_winner_table.csv')
df6 = pd.read_csv('bat_prediction_table.csv')
df7 = pd.read_csv('bat_final.csv')


dataframes = {'Pitcher Statistics': df1,
              'Pitch Type': df2,
              'Teams General Information': df3,
              'Teams Statistics': df4,
              'Award Winner Prediction Result': df5,
              'Career Path Prediction Input Information': df6,
              'Batter Salary Prediction Result': df7}

def get_data_object(user_selection):
    """
    For user selections, return the relevant in-memory data frame.
    """
    return dataframes[user_selection]
    


import dash_core_components as dcc




app.layout = html.Div([
    html.H4('Baseball Data Table'),
    html.Label('Please Select a Dataset:', style={'font-weight': 'bold'}),
    dcc.Dropdown(
        id='field-dropdown',
        options=[{'label': df, 'value': df} for df in dataframes],
        value='Pitcher Statistics',
        clearable=False
    ),
    


    
    dt.DataTable(
        # Initialise the rows
        rows=[{}],
        row_selectable=False,
        filterable=True,
        sortable=True,
        selected_row_indices=[],
        id='table'
    ),

	html.A(html.Button('Back to Home Page', className='three columns'),
    href='http://127.0.0.1:5000/#projects'),
	
    html.Div(id='selected-indexes'),\

], className='container')


def filter_data(value):
    if value == 'all':
        return df
    else:
        return df[df['c'] == value]

@app.callback(Output('table', 'rows'), [Input('field-dropdown', 'value')])
def update_table(user_selection):
    """
    For user selections, return the relevant table
    """
    df = get_data_object(user_selection)
    return df.to_dict('records')




app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})


if __name__ == '__main__':
    app.run_server(debug=True)
