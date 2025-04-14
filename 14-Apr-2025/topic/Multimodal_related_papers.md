# MBE-ARI: A Multimodal Dataset Mapping Bi-directional Engagement in Animal-Robot Interaction 

**Title (ZH)**: MBE-ARI：一种多模态数据集，用于映射动物与机器人互动的双向参与 

**Authors**: Ian Noronha, Advait Prasad Jawaji, Juan Camilo Soto, Jiajun An, Yan Gu, Upinder Kaur  

**Link**: [PDF](https://arxiv.org/pdf/2504.08646)  

**Abstract**: Animal-robot interaction (ARI) remains an unexplored challenge in robotics, as robots struggle to interpret the complex, multimodal communication cues of animals, such as body language, movement, and vocalizations. Unlike human-robot interaction, which benefits from established datasets and frameworks, animal-robot interaction lacks the foundational resources needed to facilitate meaningful bidirectional communication. To bridge this gap, we present the MBE-ARI (Multimodal Bidirectional Engagement in Animal-Robot Interaction), a novel multimodal dataset that captures detailed interactions between a legged robot and cows. The dataset includes synchronized RGB-D streams from multiple viewpoints, annotated with body pose and activity labels across interaction phases, offering an unprecedented level of detail for ARI research. Additionally, we introduce a full-body pose estimation model tailored for quadruped animals, capable of tracking 39 keypoints with a mean average precision (mAP) of 92.7%, outperforming existing benchmarks in animal pose estimation. The MBE-ARI dataset and our pose estimation framework lay a robust foundation for advancing research in animal-robot interaction, providing essential tools for developing perception, reasoning, and interaction frameworks needed for effective collaboration between robots and animals. The dataset and resources are publicly available at this https URL, inviting further exploration and development in this critical area. 

**Abstract (ZH)**: 多模态双向参与动物-机器人交互（Multimodal Bidirectional Engagement in Animal-Robot Interaction） 

---
# Training-free Guidance in Text-to-Video Generation via Multimodal Planning and Structured Noise Initialization 

**Title (ZH)**: 基于多模态规划和结构化噪声初始化的无需训练指导在文本到视频生成中的应用 

**Authors**: Jialu Li, Shoubin Yu, Han Lin, Jaemin Cho, Jaehong Yoon, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2504.08641)  

**Abstract**: Recent advancements in text-to-video (T2V) diffusion models have significantly enhanced the visual quality of the generated videos. However, even recent T2V models find it challenging to follow text descriptions accurately, especially when the prompt requires accurate control of spatial layouts or object trajectories. A recent line of research uses layout guidance for T2V models that require fine-tuning or iterative manipulation of the attention map during inference time. This significantly increases the memory requirement, making it difficult to adopt a large T2V model as a backbone. To address this, we introduce Video-MSG, a training-free Guidance method for T2V generation based on Multimodal planning and Structured noise initialization. Video-MSG consists of three steps, where in the first two steps, Video-MSG creates Video Sketch, a fine-grained spatio-temporal plan for the final video, specifying background, foreground, and object trajectories, in the form of draft video frames. In the last step, Video-MSG guides a downstream T2V diffusion model with Video Sketch through noise inversion and denoising. Notably, Video-MSG does not need fine-tuning or attention manipulation with additional memory during inference time, making it easier to adopt large T2V models. Video-MSG demonstrates its effectiveness in enhancing text alignment with multiple T2V backbones (VideoCrafter2 and CogVideoX-5B) on popular T2V generation benchmarks (T2VCompBench and VBench). We provide comprehensive ablation studies about noise inversion ratio, different background generators, background object detection, and foreground object segmentation. 

**Abstract (ZH)**: Recent advancements in text-to-video (T2V) diffusion models have significantly enhanced the visual quality of the generated videos. However, even recent T2V models find it challenging to follow text descriptions accurately, especially when the prompt requires accurate control of spatial layouts or object trajectories. A recent line of research uses layout guidance for T2V models that require fine-tuning or iterative manipulation of the attention map during inference time. This significantly increases the memory requirement, making it difficult to adopt a large T2V model as a backbone. To address this, we introduce Video-MSG, a training-free Guidance method for T2V generation based on Multimodal planning and Structured noise initialization.

Video-MSG consists of three steps, where in the first two steps, Video-MSG creates Video Sketch, a fine-grained spatio-temporal plan for the final video, specifying background, foreground, and object trajectories, in the form of draft video frames. In the last step, Video-MSG guides a downstream T2V diffusion model with Video Sketch through noise inversion and denoising. Notably, Video-MSG does not need fine-tuning or attention manipulation with additional memory during inference time, making it easier to adopt large T2V models. Video-MSG demonstrates its effectiveness in enhancing text alignment with multiple T2V backbones (VideoCrafter2 and CogVideoX-5B) on popular T2V generation benchmarks (T2VCompBench and VBench). We provide comprehensive ablation studies about noise inversion ratio, different background generators, background object detection, and foreground object segmentation.

标题：Video-MSG：一种基于多模态规划和结构化噪声初始化的无需训练的文本到视频生成指导方法 

---
