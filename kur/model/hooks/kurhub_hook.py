"""
Copyright 2017 Deepgram

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import logging

import urllib.request

from . import TrainingHook, PlotHook
from ...utils import prepare_json

from tempfile import NamedTemporaryFile

logger = logging.getLogger(__name__)

###############################################################################
class KurhubHook(TrainingHook):
    """ Hook for posting to Kurhub.
    """

    ###########################################################################
    @classmethod
    def get_name(cls):
        """ Returns the name of the hook.
        """
        return 'kurhub'

    ###########################################################################
    def __init__(self, uuid=None, endpoint="http://dev.kurhub.com/kur_updates", *args, **kwargs):
        """ Creates a new kurhub hook.
        """
        super().__init__(*args, **kwargs)

        self.uuid = uuid
        self.endpoint = endpoint

        with NamedTemporaryFile() as tmp_file:
            tmp_file.close()
            self.plot_name = tmp_file.name 

        self.plot_hook = PlotHook(self.plot_name)


        if uuid is None:
            raise ValueError('kurhub hook requires a "uuid" to be defined.')

    ###########################################################################
    def send_message(self, text, info=None):
        """ Sends a message to kurhub.
        """
        data = {
            'text': text,
            'uuid': self.uuid
        }

        url = self.endpoint
        data, header = prepare_json(data)
        self._submit(url, data, header)

    ###########################################################################     
    def send_plot_message(self, text, plot_string, info=None):
        """ Sends a plot message to kurhub.
        """
        data = {
            'text': text,
            'plot': plot_string,
            'uuid': self.uuid
        }

        url = self.endpoint
        data, header = prepare_json(data)
        self._submit(url, data, header)

    ###########################################################################
    def _submit(self, url, data, header):
        """ Submits a POST request to kurhub.
        """
        request = urllib.request.Request(
            url,
            data=data,
            headers=header,
            method='POST'
        )
        director = urllib.request.build_opener()

        try:
            response = director.open(request)
        except:                                 # pylint: disable=bare-except
            logger.exception('Failed to connect to kurhub. Make sure the URL '
                'and channel are correct. If the channel was newly created, '
                'it might take a little time for kurhub to catch up.')
        else:
            if response.code != 200:
                logger.error('Failed to post kurhub notification. Make sure '
                'the URL and channel are correct. If the channel was newly '
                'created, it might take a little time for kurhub to catch up.')

    ###########################################################################
    def notify(self, status, log=None, info=None):
        """ Sends the kurhub message.
        """
        
        self.plot_hook.notify(status, log, info)
        # check if plot
        plot_name = '{}.png'.format(self.plot_name)
        if os.path.isfile(plot_name):
            if os.stat(plot_name).st_size > 0:
                # upload
                with open(plot_name, 'rb') as plotfile:
                    import base64
                    encoded_string = base64.b64encode(plotfile.read()).decode('utf-8')

                ## send as post request base64
                self.send_plot_message('plot created', encoded_string)
                # delete after upload
                os.remove(plot_name)
                self.plot_name = 

        info = info or {}

        if status is TrainingHook.EPOCH_END:
            epoch = info.pop('epoch', None)
            total_epochs = info.pop('total_epochs', None)
            text = 'Finished epoch {} of {}.'.format(epoch, total_epochs)
        elif status is TrainingHook.TRAINING_END:
            text = 'Training has ended.'
        elif status is TrainingHook.TRAINING_START:
            text = 'Started training.'
        else:
            text = None

        if text:
            self.send_message(text, info)

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF
