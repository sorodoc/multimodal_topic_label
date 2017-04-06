This repository contains the source code and the dataset used for the paper 'Multimodal topic labelling' which can be found at http://aclweb.org/anthology/E/E17/E17-2111.pdf

The folder 'code' contains the source code for the models presented in the paper.
	- concatenation_old_models.py and model_concatenation.py are the files used for the baseline model : the model which combines feature-based textual labelling and a neural network for visual labelling;
	- dnn_separate.py and multimodal_model_separate.py contain the model with 2 separate neural networks which don't share any layers between them. It is referred as disjoint-nn in the paper.
	- dnn_shared.py and multimodal_model_shared.py contain the model with 2 neural networks for topic labelling and these networks share the last 3 layers. It is referred as joint-nn in the paper. 
	- the other files in this folder are used for preprocessing and the code is designed to work with predefined folds. For further details contact Sorodoc Ionut-Teodor.

The folder 'data' contains the data and the annotations used as input for the network.
	- textual_annotation.csv contains the user annotations for each textual label relative to a topic.
	- dataset_images.txt contains the annotation score for an image relative to a topic : it is a three column file : topic_id, image_url, annotation_score. The images are not added to this repository, but they can be sent, if somebody requires them.
	- dataset_text.txt contains the annotation score for a textual label relative to a topic : it is a three column file : topic_id, textual_label, annotation_score.
	- topics.csv contains the information about 228 topics.
