# Surformer v2: A Multimodal Classifier for Surface Understanding from Touch and Vision 

**Title (ZH)**: Surformer v2：一种基于触觉和视觉的表面理解多模态分类器 

**Authors**: Manish Kansana, Sindhuja Penchala, Shahram Rahimi, Noorbakhsh Amiri Golilarz  

**Link**: [PDF](https://arxiv.org/pdf/2509.04658)  

**Abstract**: Multimodal surface material classification plays a critical role in advancing tactile perception for robotic manipulation and interaction. In this paper, we present Surformer v2, an enhanced multi-modal classification architecture designed to integrate visual and tactile sensory streams through a late(decision level) fusion mechanism. Building on our earlier Surformer v1 framework [1], which employed handcrafted feature extraction followed by mid-level fusion architecture with multi-head cross-attention layers, Surformer v2 integrates the feature extraction process within the model itself and shifts to late fusion. The vision branch leverages a CNN-based classifier(Efficient V-Net), while the tactile branch employs an encoder-only transformer model, allowing each modality to extract modality-specific features optimized for classification. Rather than merging feature maps, the model performs decision-level fusion by combining the output logits using a learnable weighted sum, enabling adaptive emphasis on each modality depending on data context and training dynamics. We evaluate Surformer v2 on the Touch and Go dataset [2], a multi-modal benchmark comprising surface images and corresponding tactile sensor readings. Our results demonstrate that Surformer v2 performs well, maintaining competitive inference speed, suitable for real-time robotic applications. These findings underscore the effectiveness of decision-level fusion and transformer-based tactile modeling for enhancing surface understanding in multi-modal robotic perception. 

**Abstract (ZH)**: 多模态表面材料分类在提升机器人操纵和交互中的触觉感知方面起着关键作用。本文介绍了一种增强的多模态分类架构Surformer v2，该架构通过后期决策级融合机制整合视觉和触觉传感流。基于我们之前提出的Surformer v1框架，Surformer v2将特征提取过程集成到模型中，并转向后期融合。视觉分支采用基于CNN的分类器（Efficient V-Net），触觉分支采用仅编码器变压器模型，使得每种模态能够提取适用于分类的专用特征。模型通过可学习的加权和结合输出logits进行决策级融合，从而根据不同数据上下文和训练动力学灵活强调每种模态。我们在Touch and Go数据集上评估了Surformer v2，该数据集包含多模态基准中的表面图像和相应的触觉传感器读数。实验结果表明，Surformer v2表现良好，保持了竞争性的推理速度，适用于实时机器人应用。这些发现强调了决策级融合和基于变压器的触觉建模在提升多模态机器人感知中表面理解方面的有效性。

标题：
Surformer v2: Enhanced Multi-modal Classification Architecture for Tactile Perception in Robotic Manipulation 

---
# REMOTE: A Unified Multimodal Relation Extraction Framework with Multilevel Optimal Transport and Mixture-of-Experts 

**Title (ZH)**: REMOTE：一种基于多级最优传输和专家混合的统一多模态关系提取框架 

**Authors**: Xinkui Lin, Yongxiu Xu, Minghao Tang, Shilong Zhang, Hongbo Xu, Hao Xu, Yubin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.04844)  

**Abstract**: Multimodal relation extraction (MRE) is a crucial task in the fields of Knowledge Graph and Multimedia, playing a pivotal role in multimodal knowledge graph construction. However, existing methods are typically limited to extracting a single type of relational triplet, which restricts their ability to extract triplets beyond the specified types. Directly combining these methods fails to capture dynamic cross-modal interactions and introduces significant computational redundancy. Therefore, we propose a novel \textit{unified multimodal Relation Extraction framework with Multilevel Optimal Transport and mixture-of-Experts}, termed REMOTE, which can simultaneously extract intra-modal and inter-modal relations between textual entities and visual objects. To dynamically select optimal interaction features for different types of relational triplets, we introduce mixture-of-experts mechanism, ensuring the most relevant modality information is utilized. Additionally, considering that the inherent property of multilayer sequential encoding in existing encoders often leads to the loss of low-level information, we adopt a multilevel optimal transport fusion module to preserve low-level features while maintaining multilayer encoding, yielding more expressive representations. Correspondingly, we also create a Unified Multimodal Relation Extraction (UMRE) dataset to evaluate the effectiveness of our framework, encompassing diverse cases where the head and tail entities can originate from either text or image. Extensive experiments show that REMOTE effectively extracts various types of relational triplets and achieves state-of-the-art performanc on almost all metrics across two other public MRE datasets. We release our resources at this https URL. 

**Abstract (ZH)**: 统一多模态关系提取框架：多层次最优运输与专家混合方法（REMOTE） 

---
