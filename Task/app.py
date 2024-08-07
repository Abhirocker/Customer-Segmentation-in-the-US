from dash import dcc, html
from jupyter_dash import JupyterDash
import layout
import callback

app = JupyterDash(__name__)
print("app type:", type(app))

# Set the layout from the layout module
app.layout = layout.layout

# Register the callbacks
callback.register_callbacks(app)

if __name__ == '__main__':
    app.run_server(host="0.0.0.0", mode="external")