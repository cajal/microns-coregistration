import slack
import slack.errors
import traceback
import os

default_channel = '#platinum_tracer_notifications' # What channel will the widget stuff use?


class SlackForWidget(slack.WebClient):
    """
    Usage:
    ```python
    # Instantiate the client (it by default gets the token from the SLACK_BOT_TOKEN environment variable)
    slack_client = SlackForWidget()
    # Send to the default channel (or pass in a different channel that the bot has to be allowed to post in)
    slack_client.post_to_slack("sample text")
    # OR send to a user
    slack_client.send_direct_message("sample_username", "sample text")
    ```
    """
    
    def __init__(self, token=None, default_channel=default_channel):
        if token is None:
            token = os.environ.get('SLACK_BOT_TOKEN')
        super().__init__(token=token)
        self.default_channel = default_channel
        
    def post_to_slack(self, text, channel=None, as_file=False):
        if channel is None:
            channel = self.default_channel
        try:
            if as_file:
                response = self.files_upload(channels=channel, content=text)
            else:
                response = self.chat_postMessage(channel=channel, text=text)
            return response
        except slack.errors.SlackApiError:
            tb1 = traceback.format_exc()
            try:
                # send_direct_message('cpapadop', 'Slack messenger failure:\n' + str(e))
                self.files_upload(channel='@cpapadop', content=(f'Slack messenger failure for message\n:({text})\nException:\n' + tb1))
            except Exception as ee:
                tb2 = traceback.format_exc()
                print('Slack messenger failure:\n' + tb1)
                print('Slack messenger DOUBLE failure:\n' + tb2)
            return False

    def send_direct_message(self, text, slack_username , as_file=False):
        return self.post_to_slack(text, f'@{slack_username}', as_file=as_file)

    def get_slack_username(self, display_name):
        try:
            response = self.users_list()
            users_list = response['members']
            for slack_user in users_list:
                name = slack_user['name']
                if (name == display_name) or (slack_user.get('real_name') == display_name) or (slack_user['profile'].get('display_name') == display_name):
                    return name
        except slack.errors.SlackApiError:
            tb_msg = 'Get slack username failure:\n' + traceback.format_exc()
            # self.post_to_slack(tb_msg, channel='@cpapadop')
            print(tb_msg)

    def post_to_slack_and_user(self, text, slack_username, channel=None, as_file=False):
        if channel is None:
            channel = self.default_channel
        response1 = self.post_to_slack(text, channel=channel, as_file=as_file)
        response2 = self.send_direct_message(text, slack_username, as_file=as_file)
        return [response1, response2]