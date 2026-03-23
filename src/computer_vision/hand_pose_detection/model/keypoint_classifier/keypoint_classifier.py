#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np


class KeyPointClassifier(object):
    def __init__(
        self,
        model_path='model/keypoint_classifier/keypoint_classifier_v1.tflite',
        num_threads=1,
    ):
        self.model_path = os.path.abspath(model_path)
        self._printed_shape = False

        try:
            import tensorflow as tf
            self.interpreter = tf.lite.Interpreter(
                model_path=self.model_path,
                num_threads=num_threads,
            )
        except ImportError:
            try:
                from ai_edge_litert.interpreter import Interpreter
                self.interpreter = Interpreter(
                    model_path=self.model_path,
                    num_threads=num_threads,
                )
            except ImportError as e:
                raise ImportError(
                    'Neither tensorflow nor ai_edge_litert is installed.'
                ) from e

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        in_shape = self.input_details[0]['shape']
        self.expected_feat = int(in_shape[-1]) if len(in_shape) > 0 else 0

    def _adapt_input(self, landmark_list):
        """
        Match the Kotlin behavior: inspect the model input shape and copy the
        provided features into a [1, expected_feat] input buffer.
        """
        src = np.asarray(landmark_list, dtype=np.float32).flatten()
        dst = np.zeros((1, self.expected_feat), dtype=np.float32)
        copy_len = min(self.expected_feat, src.size)
        if copy_len > 0:
            dst[0, :copy_len] = src[:copy_len]

        if not self._printed_shape:
            print(
                f"KeyPointClassifier loaded: {self.model_path}\n"
                f"  model expects {self.expected_feat} features, got {src.size} from preprocessing"
            )
            self._printed_shape = True

        return dst

    def __call__(
        self,
        landmark_list,
    ):
        input_details_tensor_index = self.input_details[0]['index']
        input_tensor = self._adapt_input(landmark_list)
        self.interpreter.set_tensor(input_details_tensor_index, input_tensor)
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]['index']
        result = self.interpreter.get_tensor(output_details_tensor_index)
        result_index = int(np.argmax(np.squeeze(result)))
        return result_index
