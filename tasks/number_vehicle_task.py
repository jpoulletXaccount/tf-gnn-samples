
import tensorflow as tf
import numpy as np
from dpu_utils.utils import RichPath, LocalPath
from collections import namedtuple


from typing import Any, Dict, Tuple, List, Iterable,Iterator
from .sparse_graph_task import Sparse_Graph_Task,DataFold,MinibatchData
from utils.number_vehicles import created_dataset_utils


StopsData = namedtuple('StopsData', ['adj_lists','type_to_node_to_num_incoming_edges', 'num_stops', 'node_features', 'label'])


class Nb_Vehicles_Task(Sparse_Graph_Task):
    """
    Instancie une task de classification en nombre de vehicles
    """

    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        # Things that will be filled once we load data:
        self.__num_edge_types = 10
        self.__initial_node_feature_size = 0
        self.__num_output_classes = 5

        # specific map from taks to helpers
        self._mapping = {'created_dataset': created_dataset_utils.CreatedDatasetUtils}


    @classmethod
    def default_params(cls):
        """
        Applied to the class object, return the a list of specific param
        :return:
        """
        params = super().default_params()
        params.update({
            'add_self_loop_edges': True,
            'use_graph': True,
            'activation_function': "tanh",
            'out_layer_dropout_keep_prob': 1.0,
        })
        return params

    @staticmethod
    def name() -> str:
        return "Nb_Vehicles"

    @staticmethod
    def default_data_path() -> str:
        return "data/number_vehicles"


    def get_metadata(self) -> Dict[str, Any]:
        """
        :return: a dict with all the params related to the task
        """
        metadata = super().get_metadata()
        metadata['initial_node_feature_size'] = self.__initial_node_feature_size
        metadata['num_output_classes'] = self.__num_output_classes
        metadata['num_edge_types'] = self.__num_edge_types
        return metadata

    def restore_from_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        From a dict of parameters, restore it
        :param metadata:
        """
        super().restore_from_metadata(metadata)
        self.__initial_node_feature_size = metadata['initial_node_feature_size']
        self.__num_output_classes = metadata['num_output_classes']
        self.__num_edge_types = metadata['num_edge_types']

    @property
    def num_edge_types(self) -> int:
        return self.__num_edge_types

    @property
    def initial_node_feature_size(self) -> int:
        return self.__initial_node_feature_size

    # -------------------- Data Loading --------------------
    def load_data(self, path: RichPath) -> None:
        """
        Main function to load training and validation data
        :param path: the path to load the data
        """
        train_data, valid_data, test_data = self.__load_data(path)
        self._loaded_data[DataFold.TRAIN] = train_data
        self._loaded_data[DataFold.VALIDATION] = valid_data
        self._loaded_data[DataFold.TEST] = test_data


    def __load_data(self, data_directory: RichPath):
        assert isinstance(data_directory, LocalPath), "NumberVehiclesTask can only handle local data"
        data_path = data_directory.path
        print(" Loading NumberVehicles data from %s." % (data_path,))
        helper_loader = self._mapping[self.params['data_kind']](data_path,self.num_edge_types)
        all_dist_matrix,all_type_num, all_features, all_labels = helper_loader.load_data()
        self.__initial_node_feature_size = helper_loader.number_features
        self.__num_output_classes = helper_loader.number_labels


        train_data = self._process_raw_data(all_dist_matrix[DataFold.TRAIN],all_type_num[DataFold.TRAIN],all_features[DataFold.TRAIN],all_labels[DataFold.TRAIN])
        valid_data = self._process_raw_data(all_dist_matrix[DataFold.VALIDATION],all_type_num[DataFold.VALIDATION],all_features[DataFold.VALIDATION],all_labels[DataFold.VALIDATION])
        test_data = self._process_raw_data(all_dist_matrix[DataFold.TEST],all_type_num[DataFold.TEST],all_features[DataFold.TEST],all_labels[DataFold.TEST])

        return train_data, valid_data, test_data

    def _process_raw_data(self,dist_matrix,type_num, features, labels):
        """
        Process the data to put it into right format
        :return: data under the form of lists of StopData
        """
        processed_data = []
        for i in range(0,len(features)):
            processed_data.append(StopsData(adj_lists=dist_matrix[i],
                                            type_to_node_to_num_incoming_edges=type_num[i],
                                            num_stops=len(features[i]),
                                            node_features=features[i],
                                            label=labels[i]))

        return processed_data


    def make_task_output_model(self,
                               placeholders: Dict[str, tf.Tensor],
                               model_ops: Dict[str, tf.Tensor],
                               ) -> None:
        """
        Create task-specific output model. For this, additional placeholders
        can be created, but will need to be filled in the
        make_minibatch_iterator implementation.

        This method may assume existence of the placeholders and ops created in
        make_task_input_model and of the following:
            model_ops['final_node_representations']: a float32 tensor of shape
                [V, D], which holds the final node representations after the
                GNN layers.
            placeholders['num_graphs']: a int32 scalar holding the number of
                graphs in this batch.
        Order of nodes is preserved across all tensors.

        This method has to define model_ops['task_metrics'] to a dictionary,
        from which model_ops['task_metrics']['loss'] will be used for
        optimization. Other entries may hold additional metrics (accuracy,
        MAE, ...).

        Arguments:
            placeholders: Dictionary of placeholders used by the model,
                pre-populated by the generic graph model values, and to
                be extended with task-specific placeholders.
            model_ops: Dictionary of named operations in the model,
                pre-populated by the generic graph model values, and to
                be extended with task-specific operations.
        """
        placeholders['labels'] = tf.placeholder(tf.int32,shape=[None], name='labels')

        placeholders['graph_nodes_list'] = \
            tf.placeholder(dtype=tf.int32, shape=[None], name='graph_nodes_list')

        placeholders['out_layer_dropout_keep_prob'] =\
            tf.placeholder_with_default(input=tf.constant(1.0, dtype=tf.float32),
                                        shape=[],
                                        name='out_layer_dropout_keep_prob')

        final_node_representations = \
            tf.nn.dropout(model_ops['final_node_representations'],
                          rate=1.0 - placeholders['out_layer_dropout_keep_prob'])
        output_label_logits = \
            tf.keras.layers.Dense(units=self.__num_output_classes,
                                  use_bias=False,
                                  activation=None,
                                  name="OutputDenseLayer",
                                  )(final_node_representations)  # Shape [nb_node, Classes]

        # Sum up all nodes per-graph
        per_graph_outputs = tf.unsorted_segment_sum(data=output_label_logits,
                                                    segment_ids=placeholders['graph_nodes_list'],
                                                    num_segments=placeholders['num_graphs'])

        correct_preds = tf.equal(tf.argmax(per_graph_outputs, axis=1, output_type=tf.int32),
                                 placeholders['labels'])

        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=per_graph_outputs,
                                                                labels=placeholders['labels'])

        total_loss = tf.reduce_sum(losses)

        number_correct_preds = tf.reduce_sum(tf.cast(correct_preds,tf.float32))
        number_of_predictions = tf.cast(placeholders['num_graphs'],tf.float32)
        accuracy = number_correct_preds / number_of_predictions
        tf.summary.scalar('accuracy', accuracy)

        model_ops['task_metrics'] = {
            'loss': total_loss / number_of_predictions,
            'total_loss': total_loss,
            'accuracy': accuracy,
        }


    def make_minibatch_iterator(self,
                                data: Iterable[Any],
                                data_fold: DataFold,
                                model_placeholders: Dict[str, tf.Tensor],
                                max_nodes_per_batch: int,
                                ) -> Iterator[MinibatchData]:
        """
        Create minibatches for a sparse graph model, usually by flattening
        many smaller graphs into one large graphs of disconnected components.
        This should produce one epoch's worth of minibatches.

        Arguments:
            data: Data to iterate over, created by either load_data or
                load_eval_data_from_path.
            data_fold: Fold of the loaded data to iterate over.
            model_placeholders: The placeholders of the model that need to be
                filled with data. Aside from the placeholders introduced by the
                task in make_task_input_model and make_task_output_model.
            max_nodes_per_batch: Maximal number of nodes that can be packed
                into one batch.

        Returns:
            Iterator over MinibatchData values, which provide feed dicts
            as well as some batch statistics.
        """
        if data_fold == DataFold.TRAIN:
            np.random.shuffle(data)
            out_layer_dropout_keep_prob = self.params['out_layer_dropout_keep_prob']
        else:
            out_layer_dropout_keep_prob = 1.0

        # Pack until we cannot fit more graphs in the batch
        num_graphs = 0
        while num_graphs < len(data):
            num_graphs_in_batch = 0
            batch_node_features = []
            batch_node_labels = []
            batch_adjacency_lists = [[] for _ in range(self.num_edge_types)]
            batch_type_to_num_incoming_edges = []
            batch_graph_nodes_list = []
            node_offset = 0

            while num_graphs < len(data) and node_offset + len(data[num_graphs].node_features) < max_nodes_per_batch:
                cur_graph = data[num_graphs]
                num_nodes_in_graph = len(cur_graph.node_features)
                batch_node_features.extend(cur_graph.node_features)
                batch_graph_nodes_list.append(np.full(shape=[num_nodes_in_graph],
                                                      fill_value=num_graphs_in_batch,
                                                      dtype=np.int32))

                for i in range(self.num_edge_types):
                    batch_adjacency_lists[i].append(cur_graph.adj_lists[i] + node_offset)

                batch_type_to_num_incoming_edges.append(np.array(cur_graph.type_to_node_to_num_incoming_edges))
                batch_node_labels.append(cur_graph.label)
                num_graphs += 1
                num_graphs_in_batch += 1
                node_offset += num_nodes_in_graph


            batch_feed_dict = {
                model_placeholders['initial_node_features']: np.array(batch_node_features),
                model_placeholders['type_to_num_incoming_edges']: np.concatenate(batch_type_to_num_incoming_edges, axis=1),
                model_placeholders['graph_nodes_list']: np.concatenate(batch_graph_nodes_list),
                model_placeholders['labels']: np.array(batch_node_labels),
                model_placeholders['out_layer_dropout_keep_prob']: out_layer_dropout_keep_prob,
            }

            # Merge adjacency lists:
            num_edges = 0
            for i in range(self.num_edge_types):
                if len(batch_adjacency_lists[i]) > 0:
                    adj_list = np.concatenate(batch_adjacency_lists[i])
                else:
                    adj_list = np.zeros((0, 2), dtype=np.int32)
                num_edges += adj_list.shape[0]
                batch_feed_dict[model_placeholders['adjacency_lists'][i]] = adj_list


            yield MinibatchData(feed_dict=batch_feed_dict,
                                num_graphs=num_graphs_in_batch,
                                num_nodes=node_offset,
                                num_edges=num_edges)

    # def make_minibatch_iterator(self,
    #                             data: Iterable[Any],
    #                             data_fold: DataFold,
    #                             model_placeholders: Dict[str, tf.Tensor],
    #                             max_nodes_per_batch: int,
    #                             ) -> Iterator[MinibatchData]:
    #     """
    #     Create minibatches for a sparse graph model, usually by flattening
    #     many smaller graphs into one large graphs of disconnected components.
    #     This should produce one epoch's worth of minibatches.
    #
    #     Arguments:
    #         data: Data to iterate over, created by either load_data or
    #             load_eval_data_from_path.
    #         data_fold: Fold of the loaded data to iterate over.
    #         model_placeholders: The placeholders of the model that need to be
    #             filled with data. Aside from the placeholders introduced by the
    #             task in make_task_input_model and make_task_output_model.
    #         max_nodes_per_batch: Maximal number of nodes that can be packed
    #             into one batch.
    #
    #     Returns:
    #         Iterator over MinibatchData values, which provide feed dicts
    #         as well as some batch statistics.
    #     """
    #     if data_fold == DataFold.TRAIN:
    #         np.random.shuffle(data)
    #         out_layer_dropout_keep_prob = self.params['out_layer_dropout_keep_prob']
    #     else:
    #         out_layer_dropout_keep_prob = 1.0
    #
    #     # Pack until we cannot fit more graphs in the batch
    #     for cur_graph in data:
    #         batch_feed_dict = {
    #             model_placeholders['initial_node_features']: np.array(cur_graph.node_features),
    #             model_placeholders['type_to_num_incoming_edges']: np.array(cur_graph.type_to_node_to_num_incoming_edges),
    #             # model_placeholders['graph_nodes_list']: np.concatenate(batch_graph_nodes_list),
    #             model_placeholders['labels']: np.expand_dims(cur_graph.label,axis=0),
    #             model_placeholders['out_layer_dropout_keep_prob']: out_layer_dropout_keep_prob,
    #         }
    #
    #         # Merge adjacency lists:
    #         num_edges = 0
    #         for i in range(self.num_edge_types):
    #             if len(cur_graph.adj_lists[i]) > 0:
    #                 adj_list = cur_graph.adj_lists[i]
    #             else:
    #                 adj_list = np.zeros((0, 2), dtype=np.int32)
    #
    #             batch_feed_dict[model_placeholders['adjacency_lists'][i]] = adj_list
    #             num_edges += adj_list.shape[0]
    #
    #         yield MinibatchData(feed_dict=batch_feed_dict,
    #                             num_graphs=1,
    #                             num_nodes=len(cur_graph.node_features),
    #                             num_edges=num_edges)


    def early_stopping_metric(self,
                              task_metric_results: List[Dict[str, np.ndarray]],
                              num_graphs: int,
                              ) -> float:
        """
        Given the results of the task's metric for all minibatches of an
        epoch, produce a metric that should go down (e.g., loss). This is used
        for early stopping of training.

        Arguments:
            task_metric_results: List of the values of model_ops['task_metrics']
                (defined in make_task_model) for each of the minibatches produced
                by make_minibatch_iterator.
            num_graphs: Number of graphs processed in this epoch.

        Returns:
            Numeric value, where a lower value indicates more desirable results.
        """
        # Early stopping based on average loss:
        return np.sum([m['total_loss'] for m in task_metric_results]) / num_graphs

    def pretty_print_epoch_task_metrics(self,
                                        task_metric_results: List[Dict[str, np.ndarray]],
                                        num_graphs: int,
                                        ) -> str:
        """
        Given the results of the task's metric for all minibatches of an
        epoch, produce a human-readable result for the epoch (e.g., average
        accuracy).

        Arguments:
            task_metric_results: List of the values of model_ops['task_metrics']
                (defined in make_task_model) for each of the minibatches produced
                by make_minibatch_iterator.
            num_graphs: Number of graphs processed in this epoch.

        Returns:
            String representation of the task-specific metrics for this epoch,
            e.g., mean absolute error for a regression task.
        """
        print("length of the metric ", len(task_metric_results))
        return "Acc: %.2f%%" % (np.mean([task_metric_results[i]['accuracy'] for i in range(0,len(task_metric_results))])  * 100,)
