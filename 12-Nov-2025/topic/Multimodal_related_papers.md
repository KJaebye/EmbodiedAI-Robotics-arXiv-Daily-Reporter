# The One Where They Brain-Tune for Social Cognition: Multi-Modal Brain-Tuning on Friends 

**Title (ZH)**: 他们在其中进行脑调控以提升社会认知： FRIENDS的多模态脑调控 

**Authors**: Nico Policzer, Cameron Braunstein, Mariya Toneva  

**Link**: [PDF](https://arxiv.org/pdf/2511.07988)  

**Abstract**: Recent studies on audio models show brain-tuning - fine-tuning models to better predict corresponding fMRI activity - improves brain alignment and increases performance on downstream semantic and audio tasks. We extend this approach to a multimodal audio-video model to enhance social cognition, targeting the Superior Temporal Sulcus (STS), a key region for social processing, while subjects watch Friends. We find significant increases in brain alignment to the STS and an adjacent ROI, as well as improvements to a social cognition task related to the training data - sarcasm detection in sitcoms. In summary, our study extends brain-tuning to the multi-modal domain, demonstrating improvements to a downstream task after tuning to a relevant functional region. 

**Abstract (ZH)**: 近期关于音频模型的研究表明，通过细调模型以更好地预测相应的fMRI活动来进行脑部调优可以提高脑部对齐并增强下游语义和音频任务的性能。我们将此方法扩展到多模态音频-视频模型，旨在增强社会认知能力，针对社会处理的关键区域优越颞沟(STS)，当受试者观看《 Friends》时。我们发现大脑对齐度STS及其相邻ROI有显著提升，并且在与训练数据相关的社会认知任务（情景喜剧中的讽刺检测）上取得了改进。总之，我们的研究将脑部调优扩展到多模态领域，展示了在相关功能区域调优后对下游任务的改善。 

---
# ImagebindDC: Compressing Multi-modal Data with Imagebind-based Condensation 

**Title (ZH)**: ImagebindDC：基于Imagebind的多模态数据凝练压缩 

**Authors**: Yue Min, Shaobo Wang, Jiaze Li, Tianle Niu, Junxin Fan, Yongliang Miao, Lijin Yang, Linfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.08263)  

**Abstract**: Data condensation techniques aim to synthesize a compact dataset from a larger one to enable efficient model training, yet while successful in unimodal settings, they often fail in multimodal scenarios where preserving intricate inter-modal dependencies is crucial. To address this, we introduce ImageBindDC, a novel data condensation framework operating within the unified feature space of ImageBind. Our approach moves beyond conventional distribution-matching by employing a powerful Characteristic Function (CF) loss, which operates in the Fourier domain to facilitate a more precise statistical alignment via exact infinite moment matching. We design our objective to enforce three critical levels of distributional consistency: (i) uni-modal alignment, which matches the statistical properties of synthetic and real data within each modality; (ii) cross-modal alignment, which preserves pairwise semantics by matching the distributions of hybrid real-synthetic data pairs; and (iii) joint-modal alignment, which captures the complete multivariate data structure by aligning the joint distribution of real data pairs with their synthetic counterparts. Extensive experiments highlight the effectiveness of ImageBindDC: on the NYU-v2 dataset, a model trained on just 5 condensed datapoints per class achieves lossless performance comparable to one trained on the full dataset, achieving a new state-of-the-art with an 8.2\% absolute improvement over the previous best method and more than 4$\times$ less condensation time. 

**Abstract (ZH)**: ImageBindDC：一种基于ImageBind统一特征空间的新型数据凝练框架 

---
# Remodeling Semantic Relationships in Vision-Language Fine-Tuning 

**Title (ZH)**: 重塑视觉-语言微调中的语义关系 

**Authors**: Xiangyang Wu, Liu Liu, Baosheng Yu, Jiayan Qiu, Zhenwei Shi  

**Link**: [PDF](https://arxiv.org/pdf/2511.08238)  

**Abstract**: Vision-language fine-tuning has emerged as an efficient paradigm for constructing multimodal foundation models. While textual context often highlights semantic relationships within an image, existing fine-tuning methods typically overlook this information when aligning vision and language, thus leading to suboptimal performance. Toward solving this problem, we propose a method that can improve multimodal alignment and fusion based on both semantics and this http URL, we first extract multilevel semantic features from different vision encoder to capture more visual cues of the relationships. Then, we learn to project the vision features to group related semantics, among which are more likely to have relationships. Finally, we fuse the visual features with the textual by using inheritable cross-attention, where we globally remove the redundant visual relationships by discarding visual-language feature pairs with low correlation. We evaluate our proposed method on eight foundation models and two downstream tasks, visual question answering and image captioning, and show that it outperforms all existing methods. 

**Abstract (ZH)**: 视觉语言细调已成为构建多模态基础模型的有效范式。虽然文本上下文通常能够突出图像中的语义关系，但现有细调方法在对齐视觉和语言时往往忽视了这种信息，导致性能不佳。为了解决这一问题，我们提出了一种基于语义和此httpURL的方法，以提高多模态对齐和融合。首先，我们从不同的视觉编码器中提取多层次语义特征以捕捉更多的视觉线索。然后，学习将视觉特征投影到相关语义组中，其中更有可能存在语义关系。最后，我们通过继承性的跨注意力机制将视觉特征与文本融合，在此过程中全局移除冗余的视觉关系，通过丢弃低相关性的视觉-语言特征对。我们在八个基础模型和两个下游任务（视觉问答和图像 captioning）上评估了我们提出的方法，并证明它优于所有现有方法。 

---
# Multi-modal Deepfake Detection and Localization with FPN-Transformer 

**Title (ZH)**: 多模态深伪检测与定位：FPN-Transformer 方法 

**Authors**: Chende Zheng, Ruiqi Suo, Zhoulin Ji, Jingyi Deng, Fangbin Yi, Chenhao Lin, Chao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2511.08031)  

**Abstract**: The rapid advancement of generative adversarial networks (GANs) and diffusion models has enabled the creation of highly realistic deepfake content, posing significant threats to digital trust across audio-visual domains. While unimodal detection methods have shown progress in identifying synthetic media, their inability to leverage cross-modal correlations and precisely localize forged segments limits their practicality against sophisticated, fine-grained manipulations. To address this, we introduce a multi-modal deepfake detection and localization framework based on a Feature Pyramid-Transformer (FPN-Transformer), addressing critical gaps in cross-modal generalization and temporal boundary regression. The proposed approach utilizes pre-trained self-supervised models (WavLM for audio, CLIP for video) to extract hierarchical temporal features. A multi-scale feature pyramid is constructed through R-TLM blocks with localized attention mechanisms, enabling joint analysis of cross-context temporal dependencies. The dual-branch prediction head simultaneously predicts forgery probabilities and refines temporal offsets of manipulated segments, achieving frame-level localization precision. We evaluate our approach on the test set of the IJCAI'25 DDL-AV benchmark, showing a good performance with a final score of 0.7535 for cross-modal deepfake detection and localization in challenging environments. Experimental results confirm the effectiveness of our approach and provide a novel way for generalized deepfake detection. Our code is available at this https URL 

**Abstract (ZH)**: 生成对抗网络（GANs）和扩散模型的迅速发展使得高度逼真的深度合成内容得以创建，对音频-视觉领域的数字信任构成了重大威胁。虽然单模态检测方法在识别合成媒体方面取得了进步，但它们无法利用跨模态的关联性和精确定位伪造段落的限制，限制了其在复杂、细粒度操纵方面的实用性。为了解决这个问题，我们提出了一种基于特征金字塔变换器（FPN-Transformer）的多模态深度合成检测与定位框架，以解决跨模态泛化和时间边界回归的关键缺口。该方法利用预训练的自监督模型（WavLM用于音频，CLIP用于视频）提取分层时序特征。通过局部注意力机制的R-TLM块构建多尺度特征金字塔，实现跨上下文时序依赖性的联合分析。双重分支预测头同时预测伪造概率并精修处理段落的时间偏移，实现了帧级定位精度。我们在IJCAI'25 DDL-AV基准测试集上评估了该方法，在具有挑战性的环境中跨模态深度合成检测与定位方面取得了良好的表现，最终得分为0.7535。实验结果证实了该方法的有效性，并提供了一种通用深度合成检测的新途径。我们的代码可在以下链接获取。 

---
# Libra-MIL: Multimodal Prototypes Stereoscopic Infused with Task-specific Language Priors for Few-shot Whole Slide Image Classification 

**Title (ZH)**: Libra-MIL：融合任务特定语言先验的多模态原型立体化Few-shot全切片图像分类 

**Authors**: Zhenfeng Zhuang, Fangyu Zhou, Liansheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.07941)  

**Abstract**: While Large Language Models (LLMs) are emerging as a promising direction in computational pathology, the substantial computational cost of giga-pixel Whole Slide Images (WSIs) necessitates the use of Multi-Instance Learning (MIL) to enable effective modeling. A key challenge is that pathological tasks typically provide only bag-level labels, while instance-level descriptions generated by LLMs often suffer from bias due to a lack of fine-grained medical knowledge. To address this, we propose that constructing task-specific pathological entity prototypes is crucial for learning generalizable features and enhancing model interpretability. Furthermore, existing vision-language MIL methods often employ unidirectional guidance, limiting cross-modal synergy. In this paper, we introduce a novel approach, Multimodal Prototype-based Multi-Instance Learning, that promotes bidirectional interaction through a balanced information compression scheme. Specifically, we leverage a frozen LLM to generate task-specific pathological entity descriptions, which are learned as text prototypes. Concurrently, the vision branch learns instance-level prototypes to mitigate the model's reliance on redundant data. For the fusion stage, we employ the Stereoscopic Optimal Transport (SOT) algorithm, which is based on a similarity metric, thereby facilitating broader semantic alignment in a higher-dimensional space. We conduct few-shot classification and explainability experiments on three distinct cancer datasets, and the results demonstrate the superior generalization capabilities of our proposed method. 

**Abstract (ZH)**: 大规模语言模型在计算病理学中的新兴应用 necessitates 多实例学习以应对 gigapixel 全视野图像的高计算成本：基于多模态原型的多实例学习促进双向交互 

---
# Semantic-Consistent Bidirectional Contrastive Hashing for Noisy Multi-Label Cross-Modal Retrieval 

**Title (ZH)**: 语义一致的双向对比哈希方法用于含噪声多标签跨模态检索 

**Authors**: Likang Peng, Chao Su, Wenyuan Wu, Yuan Sun, Dezhong Peng, Xi Peng, Xu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.07780)  

**Abstract**: Cross-modal hashing (CMH) facilitates efficient retrieval across different modalities (e.g., image and text) by encoding data into compact binary representations. While recent methods have achieved remarkable performance, they often rely heavily on fully annotated datasets, which are costly and labor-intensive to obtain. In real-world scenarios, particularly in multi-label datasets, label noise is prevalent and severely degrades retrieval performance. Moreover, existing CMH approaches typically overlook the partial semantic overlaps inherent in multi-label data, limiting their robustness and generalization. To tackle these challenges, we propose a novel framework named Semantic-Consistent Bidirectional Contrastive Hashing (SCBCH). The framework comprises two complementary modules: (1) Cross-modal Semantic-Consistent Classification (CSCC), which leverages cross-modal semantic consistency to estimate sample reliability and reduce the impact of noisy labels; (2) Bidirectional Soft Contrastive Hashing (BSCH), which dynamically generates soft contrastive sample pairs based on multi-label semantic overlap, enabling adaptive contrastive learning between semantically similar and dissimilar samples across modalities. Extensive experiments on four widely-used cross-modal retrieval benchmarks validate the effectiveness and robustness of our method, consistently outperforming state-of-the-art approaches under noisy multi-label conditions. 

**Abstract (ZH)**: 跨模态哈希（CMH）通过将数据编码为紧凑的二进制表示，促进不同模态（例如，图像和文本）之间的高效检索。尽管近期方法取得了显著的性能，但它们通常高度依赖于完全标注的数据集，而这些数据集的获取成本高且耗时。在现实场景中，尤其是在多标签数据集中，标签噪声普遍存在，严重降低了检索性能。此外，现有的CMH方法通常忽略多标签数据中存在的部分语义重叠，限制了其鲁棒性和泛化能力。为了解决这些挑战，我们提出了一种新颖的框架，名为语义一致性双向对比哈希（SCBCH）。该框架包含两个互补的模块：（1）跨模态语义一致性分类（CSCC），利用跨模态语义一致性估计样本可靠性并减少噪声标签的影响；（2）双向软对比哈希（BSCH），基于多标签语义重叠动态生成软对比样本对，使得跨模态语义相似和不同的样本能够进行自适应对比学习。在四个广泛使用的跨模态检索基准数据集上的广泛实验验证了我们方法的有效性和鲁棒性，能够在噪声多标签条件下持续超越现有最佳方法。 

---
