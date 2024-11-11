# Self-Adaptive Image-Text Fusion for Medical Image Classification
Abstract: Multimodal classification using both medical images and text reports propels the computer aided disease diagnosis. The classification performance is susceptible to the quality of image-text fusion. However, due to the semantic gap and weak correlation between image and text modality, current image-text fusion approaches cannot achieve satisfactory results. To this end, we propose a self-adaptive image-text fusion approach to multimodal medical image classification problem. We learn a linear mapping from image to text to achieve semantic alignment that mitigates the inter-modality semantic gap, and subsequently estimate a binary correlation mask with Kullback-Leibler Divergence loss to retrieve image and text features that have strong correlations to achieve feature alignment. Then, we propose a parameter-free feature fusion method based on a pseudo-attention mechanism, which queries image features using text features and concatenates the results with image features to achieve computationally efficient feature fusion. We fuse all the image and text features for medical image classification. The experimental results on IU X-ray and COV-CTR datasets reveal that the proposed approach outperforms a group of state-of-the-art methods, and demonstrates superior medical interpretability.
# Usage
### Training image and text models:  
`python img_classify.py`
`python txt_classify.py`  
Save image and text model parameters:  
`torch.save(model.state_dict(), '/path/to/image/model.pth')`  
`torch.save(model.state_dict(), '/path/to/text/model.pth')`  
### Training a multimodal model:  
Set text and image model paths: `model_path1, model_path2 = '/path/to/text/model.pth', '/path/to/image/model.pth'`  
`python multimodal_classify.py`
