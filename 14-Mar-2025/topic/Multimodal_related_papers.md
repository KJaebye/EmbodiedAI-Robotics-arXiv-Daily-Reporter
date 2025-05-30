# DriveLMM-o1: A Step-by-Step Reasoning Dataset and Large Multimodal Model for Driving Scenario Understanding 

**Title (ZH)**: DriveLMM-o1: 一种驾驶场景理解的逐步推理数据集和大规模多模态模型 

**Authors**: Ayesha Ishaq, Jean Lahoud, Ketan More, Omkar Thawakar, Ritesh Thawkar, Dinura Dissanayake, Noor Ahsan, Yuhao Li, Fahad Shahbaz Khan, Hisham Cholakkal, Ivan Laptev, Rao Muhammad Anwer, Salman Khan  

**Link**: [PDF](https://arxiv.org/pdf/2503.10621)  

**Abstract**: While large multimodal models (LMMs) have demonstrated strong performance across various Visual Question Answering (VQA) tasks, certain challenges require complex multi-step reasoning to reach accurate answers. One particularly challenging task is autonomous driving, which demands thorough cognitive processing before decisions can be made. In this domain, a sequential and interpretive understanding of visual cues is essential for effective perception, prediction, and planning. Nevertheless, common VQA benchmarks often focus on the accuracy of the final answer while overlooking the reasoning process that enables the generation of accurate responses. Moreover, existing methods lack a comprehensive framework for evaluating step-by-step reasoning in realistic driving scenarios. To address this gap, we propose DriveLMM-o1, a new dataset and benchmark specifically designed to advance step-wise visual reasoning for autonomous driving. Our benchmark features over 18k VQA examples in the training set and more than 4k in the test set, covering diverse questions on perception, prediction, and planning, each enriched with step-by-step reasoning to ensure logical inference in autonomous driving scenarios. We further introduce a large multimodal model that is fine-tuned on our reasoning dataset, demonstrating robust performance in complex driving scenarios. In addition, we benchmark various open-source and closed-source methods on our proposed dataset, systematically comparing their reasoning capabilities for autonomous driving tasks. Our model achieves a +7.49% gain in final answer accuracy, along with a 3.62% improvement in reasoning score over the previous best open-source model. Our framework, dataset, and model are available at this https URL. 

**Abstract (ZH)**: 大型多模态模型在视觉问答任务中的逐步视觉推理研究：-driveLMM-o1数据集与基准 

---
# Dual-Stage Cross-Modal Network with Dynamic Feature Fusion for Emotional Mimicry Intensity Estimation 

**Title (ZH)**: 具有动态特征融合的双阶段跨模态网络在情感模仿强度估计中的应用 

**Authors**: Jun Yu, Lingsi Zhu, Yanjun Chi, Yunxiang Zhang, Yang Zheng, Yongqi Wang, Xilong Lu  

**Link**: [PDF](https://arxiv.org/pdf/2503.10603)  

**Abstract**: Emotional Mimicry Intensity (EMI) estimation serves as a critical technology for understanding human social behavior and enhancing human-computer interaction experiences, where the core challenge lies in dynamic correlation modeling and robust fusion of multimodal temporal signals. To address the limitations of existing methods in insufficient exploitation of modal synergistic effects, noise sensitivity, and limited fine-grained alignment capabilities, this paper proposes a dual-stage cross-modal alignment framework. First, we construct vision-text and audio-text contrastive learning networks based on an improved CLIP architecture, achieving preliminary alignment in the feature space through modality-decoupled pre-training. Subsequently, we design a temporal-aware dynamic fusion module that combines Temporal Convolutional Networks (TCN) and gated bidirectional LSTM to respectively capture the macro-evolution patterns of facial expressions and local dynamics of acoustic features. Innovatively, we introduce a quality-guided modality fusion strategy that enables modality compensation under occlusion and noisy scenarios through differentiable weight allocation. Experimental results on the Hume-Vidmimic2 dataset demonstrate that our method achieves an average Pearson correlation coefficient of 0.35 across six emotion dimensions, outperforming the best baseline by 40\%. Ablation studies further validate the effectiveness of the dual-stage training strategy and dynamic fusion mechanism, providing a novel technical pathway for fine-grained emotion analysis in open environments. 

**Abstract (ZH)**: 情绪模仿强度（EMI）估计作为理解人类社会行为和增强人机交互体验的关键技术，其核心挑战在于多模态时空信号的动态关联建模和鲁棒融合。为了解决现有方法在模态协同效应利用不足、对噪声敏感以及细粒度对齐能力有限的限制，本文提出了一种双阶段跨模态对齐框架。首先，我们在改进的CLIP架构基础上构建了视讯-文本和音频-文本对比学习网络，通过模态解耦预训练在特征空间中实现初步对齐。随后，我们设计了一个基于时序卷积网络（TCN）和门控双向LSTM的时序感知动态融合模块，分别捕捉面部表情的宏观演变模式和声学特征的局部动态。创新性地，我们引入了一种质量导向的模态融合策略，在遮挡和噪声场景下通过可微权重分配实现模态补偿。实验结果表明，本文方法在Hume-Vidmimic2数据集上六种情感维度上的平均皮尔森相关系数达到0.35，比最佳基线高出40%。消融研究进一步验证了双阶段训练策略和动态融合机制的有效性，为开放环境下的细粒度情感分析提供了新的技术路径。 

---
# VisualWebInstruct: Scaling up Multimodal Instruction Data through Web Search 

**Title (ZH)**: 视觉网页指令：通过网络搜索扩展多模态指令数据 

**Authors**: Yiming Jia, Jiachen Li, Xiang Yue, Bo Li, Ping Nie, Kai Zou, Wenhu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.10582)  

**Abstract**: Vision-Language Models have made significant progress on many perception-focused tasks, however, their progress on reasoning-focused tasks seem to be limited due to the lack of high-quality and diverse training data. In this work, we aim to address the scarcity issue of reasoning-focused multimodal datasets. We propose VisualWebInstruct - a novel approach that leverages search engine to create a diverse, and high-quality dataset spanning multiple disciplines like math, physics, finance, chemistry, etc. Starting with meticulously selected 30,000 seed images, we employ Google Image search to identify websites containing similar images. We collect and process the HTMLs from over 700K unique URL sources. Through a pipeline of content extraction, filtering and synthesis, we build a dataset of approximately 900K question-answer pairs, with 40% being visual QA pairs and the rest as text QA pairs. Models fine-tuned on VisualWebInstruct demonstrate significant performance gains: (1) training from Llava-OV-mid shows 10-20% absolute point gains across benchmarks, (2) training from MAmmoTH-VL shows 5% absoluate gain. Our best model MAmmoTH-VL2 shows state-of-the-art performance within the 10B parameter class on MMMU-Pro-std (40.7%), MathVerse (42.6%), and DynaMath (55.7%). These remarkable results highlight the effectiveness of our dataset in enhancing VLMs' reasoning capabilities for complex multimodal tasks. 

**Abstract (ZH)**: Vision-Language模型在许多感知任务上取得了显著进展，但在推理任务上的进展受限于高质量和多样性的训练数据不足。本工作中，我们旨在解决推理导向的多模态数据集短缺的问题。我们提出了VisualWebInstruct——一种利用搜索引擎创建多样且高质量数据集的方法，涵盖数学、物理、金融、化学等多个学科。从精心挑选的30,000张种子图像开始，我们使用Google图像搜索来识别包含相似图像的网站，并收集和处理来自超过700,000个唯一URL来源的HTML页面。通过内容提取、过滤和合成的流程，我们构建了一个大约包含900,000个问答对的数据集，其中约40%是视觉问答对，其余的是文本问答对。在VisualWebInstruct上微调的模型显示出显著的性能提升：(1) 从Llava-OV-mid训练显示出各基准上10-20%的绝对点数提升，(2) 从MAmmoTH-VL训练显示出5%的绝对点数提升。我们的最佳模型MAmmoTH-VL2在10B参数级别上，在MMMU-Pro-std（40.7%）、MathVerse（42.6%）和DynaMath（55.7%）上显示出最先进的性能。这些显著的结果突显了我们数据集在提升VLMs处理复杂多模态任务的推理能力方面的有效性。 

---
# CINEMA: Coherent Multi-Subject Video Generation via MLLM-Based Guidance 

**Title (ZH)**: CINEMA：基于MLLM的引导一致多主体视频生成 

**Authors**: Yufan Deng, Xun Guo, Yizhi Wang, Jacob Zhiyuan Fang, Angtian Wang, Shenghai Yuan, Yiding Yang, Bo Liu, Haibin Huang, Chongyang Ma  

**Link**: [PDF](https://arxiv.org/pdf/2503.10391)  

**Abstract**: Video generation has witnessed remarkable progress with the advent of deep generative models, particularly diffusion models. While existing methods excel in generating high-quality videos from text prompts or single images, personalized multi-subject video generation remains a largely unexplored challenge. This task involves synthesizing videos that incorporate multiple distinct subjects, each defined by separate reference images, while ensuring temporal and spatial consistency. Current approaches primarily rely on mapping subject images to keywords in text prompts, which introduces ambiguity and limits their ability to model subject relationships effectively. In this paper, we propose CINEMA, a novel framework for coherent multi-subject video generation by leveraging Multimodal Large Language Model (MLLM). Our approach eliminates the need for explicit correspondences between subject images and text entities, mitigating ambiguity and reducing annotation effort. By leveraging MLLM to interpret subject relationships, our method facilitates scalability, enabling the use of large and diverse datasets for training. Furthermore, our framework can be conditioned on varying numbers of subjects, offering greater flexibility in personalized content creation. Through extensive evaluations, we demonstrate that our approach significantly improves subject consistency, and overall video coherence, paving the way for advanced applications in storytelling, interactive media, and personalized video generation. 

**Abstract (ZH)**: 视频生成领域在深度生成模型，特别是扩散模型的出现下取得了显著进展。尽管现有方法在从文本提示或单张图像生成高质量视频方面表现出色，但个性化多主体视频生成仍然是一个未被充分探索的挑战。该任务涉及合成包含多个独立主体的视频，每个主体由单独的参考图像定义，同时确保时间和空间的一致性。当前方法主要依赖于将主体图像映射到文本提示中的关键词，这引入了不确定性并限制了其对主体关系建模的能力。在本文中，我们提出CINEMA，一种利用多模态大型语言模型（MLLM）进行连贯多主体视频生成的新框架。我们的方法消除了主体图像与文本实体之间需要明确对应的需要，从而降低了不确定性并减少了注释工作量。通过利用MLLM解释主体关系，我们的方法增强了可扩展性，使得可以用大量的多样化数据集进行训练。此外，我们的框架可以针对不同数量的主体进行条件化，提供了更大的个性化内容创作灵活性。通过广泛评估，我们证明了我们的方法显著提高了主体一致性并增强了整体视频连贯性，为叙事、交互式媒体和个人化视频生成等高级应用铺平了道路。 

---
# A Multimodal Fusion Model Leveraging MLP Mixer and Handcrafted Features-based Deep Learning Networks for Facial Palsy Detection 

**Title (ZH)**: 利用MLP混合器和基于手工特征的深度学习网络的多模态融合模型在面瘫检测中的应用 

**Authors**: Heng Yim Nicole Oo, Min Hun Lee, Jeong Hoon Lim  

**Link**: [PDF](https://arxiv.org/pdf/2503.10371)  

**Abstract**: Algorithmic detection of facial palsy offers the potential to improve current practices, which usually involve labor-intensive and subjective assessments by clinicians. In this paper, we present a multimodal fusion-based deep learning model that utilizes an MLP mixer-based model to process unstructured data (i.e. RGB images or images with facial line segments) and a feed-forward neural network to process structured data (i.e. facial landmark coordinates, features of facial expressions, or handcrafted features) for detecting facial palsy. We then contribute to a study to analyze the effect of different data modalities and the benefits of a multimodal fusion-based approach using videos of 20 facial palsy patients and 20 healthy subjects. Our multimodal fusion model achieved 96.00 F1, which is significantly higher than the feed-forward neural network trained on handcrafted features alone (82.80 F1) and an MLP mixer-based model trained on raw RGB images (89.00 F1). 

**Abstract (ZH)**: 基于多模态融合的深度学习模型在面部瘫痪检测中的应用：改善当前劳动密集型和主观的临床评估实践 

---
# ImageScope: Unifying Language-Guided Image Retrieval via Large Multimodal Model Collective Reasoning 

**Title (ZH)**: ImageScope: 统一语言引导的图像检索 via 大规模多模态模型联合推理 

**Authors**: Pengfei Luo, Jingbo Zhou, Tong Xu, Yuan Xia, Linli Xu, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.10166)  

**Abstract**: With the proliferation of images in online content, language-guided image retrieval (LGIR) has emerged as a research hotspot over the past decade, encompassing a variety of subtasks with diverse input forms. While the development of large multimodal models (LMMs) has significantly facilitated these tasks, existing approaches often address them in isolation, requiring the construction of separate systems for each task. This not only increases system complexity and maintenance costs, but also exacerbates challenges stemming from language ambiguity and complex image content, making it difficult for retrieval systems to provide accurate and reliable results. To this end, we propose ImageScope, a training-free, three-stage framework that leverages collective reasoning to unify LGIR tasks. The key insight behind the unification lies in the compositional nature of language, which transforms diverse LGIR tasks into a generalized text-to-image retrieval process, along with the reasoning of LMMs serving as a universal verification to refine the results. To be specific, in the first stage, we improve the robustness of the framework by synthesizing search intents across varying levels of semantic granularity using chain-of-thought (CoT) reasoning. In the second and third stages, we then reflect on retrieval results by verifying predicate propositions locally, and performing pairwise evaluations globally. Experiments conducted on six LGIR datasets demonstrate that ImageScope outperforms competitive baselines. Comprehensive evaluations and ablation studies further confirm the effectiveness of our design. 

**Abstract (ZH)**: 基于语言引导的图像检索（LGIR）任务的训练-free三阶段统一框架：ImageScope 

---
# Fine-tuning Vision Language Models with Graph-based Knowledge for Explainable Medical Image Analysis 

**Title (ZH)**: 基于图知识的 Fine-tuning 视觉语言模型以实现可解释的医学图像分析 

**Authors**: Chenjun Li, Laurin Lux, Alexander H. Berger, Martin J. Menten, Mert R. Sabuncu, Johannes C. Paetzold  

**Link**: [PDF](https://arxiv.org/pdf/2503.09808)  

**Abstract**: Accurate staging of Diabetic Retinopathy (DR) is essential for guiding timely interventions and preventing vision loss. However, current staging models are hardly interpretable, and most public datasets contain no clinical reasoning or interpretation beyond image-level labels. In this paper, we present a novel method that integrates graph representation learning with vision-language models (VLMs) to deliver explainable DR diagnosis. Our approach leverages optical coherence tomography angiography (OCTA) images by constructing biologically informed graphs that encode key retinal vascular features such as vessel morphology and spatial connectivity. A graph neural network (GNN) then performs DR staging while integrated gradients highlight critical nodes and edges and their individual features that drive the classification decisions. We collect this graph-based knowledge which attributes the model's prediction to physiological structures and their characteristics. We then transform it into textual descriptions for VLMs. We perform instruction-tuning with these textual descriptions and the corresponding image to train a student VLM. This final agent can classify the disease and explain its decision in a human interpretable way solely based on a single image input. Experimental evaluations on both proprietary and public datasets demonstrate that our method not only improves classification accuracy but also offers more clinically interpretable results. An expert study further demonstrates that our method provides more accurate diagnostic explanations and paves the way for precise localization of pathologies in OCTA images. 

**Abstract (ZH)**: 基于图表示学习和视觉语言模型的糖尿病视网膜病变解释性分级方法 

---
# Certainly Bot Or Not? Trustworthy Social Bot Detection via Robust Multi-Modal Neural Processes 

**Title (ZH)**: 一定是机器人吗？基于稳健多模态神经过程的社会机器人可信检测 

**Authors**: Qi Wu, Yingguang Yang, hao liu, Hao Peng, Buyun He, Yutong Xia, Yong Liao  

**Link**: [PDF](https://arxiv.org/pdf/2503.09626)  

**Abstract**: Social bot detection is crucial for mitigating misinformation, online manipulation, and coordinated inauthentic behavior. While existing neural network-based detectors perform well on benchmarks, they struggle with generalization due to distribution shifts across datasets and frequently produce overconfident predictions for out-of-distribution accounts beyond the training data. To address this, we introduce a novel Uncertainty Estimation for Social Bot Detection (UESBD) framework, which quantifies the predictive uncertainty of detectors beyond mere classification. For this task, we propose Robust Multi-modal Neural Processes (RMNP), which aims to enhance the robustness of multi-modal neural processes to modality inconsistencies caused by social bot camouflage. RMNP first learns unimodal representations through modality-specific encoders. Then, unimodal attentive neural processes are employed to encode the Gaussian distribution of unimodal latent variables. Furthermore, to avoid social bots stealing human features to camouflage themselves thus causing certain modalities to provide conflictive information, we introduce an evidential gating network to explicitly model the reliability of modalities. The joint latent distribution is learned through the generalized product of experts, which takes the reliability of each modality into consideration during fusion. The final prediction is obtained through Monte Carlo sampling of the joint latent distribution followed by a decoder. Experiments on three real-world benchmarks show the effectiveness of RMNP in classification and uncertainty estimation, as well as its robustness to modality conflicts. 

**Abstract (ZH)**: 社会机器人检测对于减轻错误信息、在线操控和 coordinative 不真实行为至关重要。尽管现有的基于神经网络的检测器在基准测试中表现良好，但由于数据集分布转移导致的一般化困难，它们在生成超出训练数据之外的 out-of-distribution 帐号的置信预测时往往会过于自信。为解决这一问题，我们提出了一种新的社会机器人检测中的不确定性估计框架（UESBD），该框架超越了单纯的分类预测，量化了检测器的预测不确定性。为此任务，我们提出了鲁棒多模态神经过程（RMNP），旨在增强多模态神经过程对由社会机器人伪装引起的模态不一致性鲁棒性。RMNP 首先通过模态特定编码器学习单模表示。然后，使用单模注意神经过程编码单模潜变量的高斯分布。此外，为了防止社会机器人通过伪装窃取人类特征，从而导致某些模态提供矛盾信息，我们引入了一个证据门控网络以明确建模模态的可靠性。通过广义专家产品学习联合潜分布，在融合时考虑每个模态的可靠性。最终预测通过联合潜分布的蒙特卡洛采样和解码器获得。在三个真实世界基准上的实验表明，RMNP 在分类和不确定性估计中有效，并且能够抵抗模态冲突。 

---
