# MK-Pose: Category-Level Object Pose Estimation via Multimodal-Based Keypoint Learning 

**Title (ZH)**: MK-Pose：基于多模态关键点学习的类别级对象姿态估计 

**Authors**: Yifan Yang, Peili Song, Enfan Lan, Dong Liu, Jingtai Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.06662)  

**Abstract**: Category-level object pose estimation, which predicts the pose of objects within a known category without prior knowledge of individual instances, is essential in applications like warehouse automation and manufacturing. Existing methods relying on RGB images or point cloud data often struggle with object occlusion and generalization across different instances and categories. This paper proposes a multimodal-based keypoint learning framework (MK-Pose) that integrates RGB images, point clouds, and category-level textual descriptions. The model uses a self-supervised keypoint detection module enhanced with attention-based query generation, soft heatmap matching and graph-based relational modeling. Additionally, a graph-enhanced feature fusion module is designed to integrate local geometric information and global context. MK-Pose is evaluated on CAMERA25 and REAL275 dataset, and is further tested for cross-dataset capability on HouseCat6D dataset. The results demonstrate that MK-Pose outperforms existing state-of-the-art methods in both IoU and average precision without shape priors. Codes will be released at \href{this https URL}{this https URL}. 

**Abstract (ZH)**: 基于多模态的关键点学习框架（MK-Pose）：类别级物体姿态估计 

---
# LIRA: Inferring Segmentation in Large Multi-modal Models with Local Interleaved Region Assistance 

**Title (ZH)**: LIRA：局部交织区域辅助的大规模多模态模型分割推理 

**Authors**: Zhang Li, Biao Yang, Qiang Liu, Shuo Zhang, Zhiyin Ma, Shuo Zhang, Liang Yin, Linger Deng, Yabo Sun, Yuliang Liu, Xiang Bai  

**Link**: [PDF](https://arxiv.org/pdf/2507.06272)  

**Abstract**: While large multi-modal models (LMMs) demonstrate promising capabilities in segmentation and comprehension, they still struggle with two limitations: inaccurate segmentation and hallucinated comprehension. These challenges stem primarily from constraints in weak visual comprehension and a lack of fine-grained perception. To alleviate these limitations, we propose LIRA, a framework that capitalizes on the complementary relationship between visual comprehension and segmentation via two key components: (1) Semantic-Enhanced Feature Extractor (SEFE) improves object attribute inference by fusing semantic and pixel-level features, leading to more accurate segmentation; (2) Interleaved Local Visual Coupling (ILVC) autoregressively generates local descriptions after extracting local features based on segmentation masks, offering fine-grained supervision to mitigate hallucinations. Furthermore, we find that the precision of object segmentation is positively correlated with the latent related semantics of the <seg> token. To quantify this relationship and the model's potential semantic inferring ability, we introduce the Attributes Evaluation (AttrEval) dataset. Our experiments show that LIRA achieves state-of-the-art performance in both segmentation and comprehension tasks. Code will be available at this https URL. 

**Abstract (ZH)**: 尽管大型多模态模型在分割和理解方面表现出色，但仍面临两个限制：不准确的分割和幻觉的理解。这些挑战主要源于视觉理解能力较弱和细粒度感知能力不足。为了解决这些限制，我们提出了LIRA框架，通过视觉理解与分割之间的互补关系，利用两个关键组件：(1) 语义增强特征提取器（SEFE）通过融合语义和像素级特征来提高对象属性推断，从而提高分割的准确性；(2) 交互式的局部视觉耦合（ILVC）在提取基于分割掩膜的局部特征后自回归生成局部描述，提供细粒度的监督以缓解幻觉现象。此外，我们发现对象分割的精度与<seg>标记的潜在相关语义之间的关系呈正相关。为了量化这种关系和模型的潜在语义推断能力，我们引入了属性评估数据集（AttrEval）。我们的实验表明，LIRA在分割和理解任务上均达到了最先进的性能。代码将在以下地址公开：this https URL。 

---
