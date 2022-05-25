from traitlets import Unicode, Dict, Unicode
from ipywidgets import DOMWidget, register

# CODE SNIPPET FOR FETCHING USERNAME
# put in first cell of jupyter notebook
# https://gist.github.com/danlester/ac1d5f29358ce1950482f8e7d4301f86
"""
%%javascript

require.undef('user_widget');

define('user_widget', ["@jupyter-widgets/base"], function (widgets) {

    
    var UserView = widgets.DOMWidgetView.extend({
        initialize: function(attributes, options) {

           this.response = fetch(
                '/hub/dashboards-api/hub-info/user',
                 { 
                 mode: 'no-cors', 
                 credentials: 'same-origin',
                 headers: new Headers({'Access-Control-Allow-Origin':'*'}) 
            });
            this.response.then( response => {
                this.result = response.json();
                this.result.then( json => {
                    this.model.set('value', json);
                    this.model.set('name', json.name);
                    this.model.save_changes();
                });

            });
        
        },
        
        render: async function () {
            await this.response;
            await this.result;
            var json = this.model.get('value');
            var text = 'No user';
            if (json.hasOwnProperty('name')) {
                text = 'Rendered by Javascript: '+json['name'];
            }
            this.el.appendChild(document.createTextNode(text));

        },
       
    });

    return {
        UserView: UserView
    };
});
"""

@register
class DashboardUser(DOMWidget):
    # https://gist.github.com/danlester/ac1d5f29358ce1950482f8e7d4301f86
    _view_name = Unicode('UserView').tag(sync=True)
    _view_module = Unicode('user_widget').tag(sync=True)
    _view_module_version = Unicode('0.1.0').tag(sync=True)

    value = Dict({}, help="User info").tag(sync=True)
    name = Unicode('').tag(sync=True)