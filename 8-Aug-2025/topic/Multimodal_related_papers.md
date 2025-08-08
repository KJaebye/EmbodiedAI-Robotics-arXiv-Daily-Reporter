# Analyzing the Impact of Multimodal Perception on Sample Complexity and Optimization Landscapes in Imitation Learning 

**Title (ZH)**: 分析多模态感知对模仿学习中样本复杂性和优化景观的影响 

**Authors**: Luai Abuelsamen, Temitope Lukman Adebanjo  

**Link**: [PDF](https://arxiv.org/pdf/2508.05077)  

**Abstract**: This paper examines the theoretical foundations of multimodal imitation learning through the lens of statistical learning theory. We analyze how multimodal perception (RGB-D, proprioception, language) affects sample complexity and optimization landscapes in imitation policies. Building on recent advances in multimodal learning theory, we show that properly integrated multimodal policies can achieve tighter generalization bounds and more favorable optimization landscapes than their unimodal counterparts. We provide a comprehensive review of theoretical frameworks that explain why multimodal architectures like PerAct and CLIPort achieve superior performance, connecting these empirical results to fundamental concepts in Rademacher complexity, PAC learning, and information theory. 

**Abstract (ZH)**: 本文通过统计学习理论的视角考察了多模态模仿学习的理论基础。我们分析了多模态感知（RGB-D、本体感觉、语言）对样本复杂性和模仿策略优化景观的影响。基于近期多模态学习理论的进展，我们展示出恰当整合的多模态策略可以比其单模态对应策略获得更紧的泛化界和更有利于优化的景观。我们提供了一个全面的理论框架综述，解释了诸如PerAct和CLIPort等多模态架构为何能实现优越性能，并将这些实证结果与拉德马赫复杂性、PAC学习和信息理论中的基本概念联系起来。 

---
# MV-Debate: Multi-view Agent Debate with Dynamic Reflection Gating for Multimodal Harmful Content Detection in Social Media 

**Title (ZH)**: MV-Debate：多视角代理辩论with动态反思门控用于社交媒体多模态有害内容检测 

**Authors**: Rui Lu, Jinhe Bi, Yunpu Ma, Feng Xiao, Yuntao Du, Yijun Tian  

**Link**: [PDF](https://arxiv.org/pdf/2508.05557)  

**Abstract**: Social media has evolved into a complex multimodal environment where text, images, and other signals interact to shape nuanced meanings, often concealing harmful intent. Identifying such intent, whether sarcasm, hate speech, or misinformation, remains challenging due to cross-modal contradictions, rapid cultural shifts, and subtle pragmatic cues. To address these challenges, we propose MV-Debate, a multi-view agent debate framework with dynamic reflection gating for unified multimodal harmful content detection. MV-Debate assembles four complementary debate agents, a surface analyst, a deep reasoner, a modality contrast, and a social contextualist, to analyze content from diverse interpretive perspectives. Through iterative debate and reflection, the agents refine responses under a reflection-gain criterion, ensuring both accuracy and efficiency. Experiments on three benchmark datasets demonstrate that MV-Debate significantly outperforms strong single-model and existing multi-agent debate baselines. This work highlights the promise of multi-agent debate in advancing reliable social intent detection in safety-critical online contexts. 

**Abstract (ZH)**: 社会媒体已演变成一个多模态的复杂环境，其中文本、图像和其他信号相互作用以塑造复杂的含义，常常掩盖潜在的危害意图。识别这种意图，无论是讽刺、仇恨言论还是 misinformation，由于跨模态矛盾、快速的文化变迁以及微妙的语用线索，仍然具有挑战性。为应对这些挑战，我们提出了一种名为MV-Debate的多视角代理辩论框架，该框架通过动态反思门控实现统一的多模态有害内容检测。MV-Debate结合了四位互补的辩论代理，包括表层分析师、深层推理者、模态对比和社交语境主义者，从多角度分析内容。通过迭代辩论和反思，代理在反思增益准则下精炼响应，确保准确性和高效性。在三个基准数据集上的实验表明，MV-Debate显著优于强大的单模型和现有的多代理辩论基线。这项工作强调了多代理辩论在安全关键的在线环境中促进可靠的社会意图检测方面的潜力。 

---
# StructVRM: Aligning Multimodal Reasoning with Structured and Verifiable Reward Models 

**Title (ZH)**: StructVRM: 结构化可验证奖励模型引导的多模态推理对齐 

**Authors**: Xiangxiang Zhang, Jingxuan Wei, Donghong Zhong, Qi Chen, Caijun Jia, Cheng Tan, Jinming Gu, Xiaobo Qin, Zhiping Liu, Liang Hu, Tong Sun, Yuchen Wu, Zewei Sun, Chenwei Lou, Hua Zheng, Tianyang Zhan, Changbao Wang, Shuangzhi Wu, Zefa Lin, Chang Guo, Sihang Yuan, Riwei Chen, Shixiong Zhao, Yingping Zhang, Gaowei Wu, Bihui Yu, Jiahui Wu, Zhehui Zhao, Qianqian Liu, Ruofeng Tang, Xingyue Huang, Bing Zhao, Mengyang Zhang, Youqiang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.05383)  

**Abstract**: Existing Vision-Language Models often struggle with complex, multi-question reasoning tasks where partial correctness is crucial for effective learning. Traditional reward mechanisms, which provide a single binary score for an entire response, are too coarse to guide models through intricate problems with multiple sub-parts. To address this, we introduce StructVRM, a method that aligns multimodal reasoning with Structured and Verifiable Reward Models. At its core is a model-based verifier trained to provide fine-grained, sub-question-level feedback, assessing semantic and mathematical equivalence rather than relying on rigid string matching. This allows for nuanced, partial credit scoring in previously intractable problem formats. Extensive experiments demonstrate the effectiveness of StructVRM. Our trained model, Seed-StructVRM, achieves state-of-the-art performance on six out of twelve public multimodal benchmarks and our newly curated, high-difficulty STEM-Bench. The success of StructVRM validates that training with structured, verifiable rewards is a highly effective approach for advancing the capabilities of multimodal models in complex, real-world reasoning domains. 

**Abstract (ZH)**: 现有的视觉-语言模型在复杂的多问题推理任务中往往表现不佳，特别是在需要部分正确性以实现有效学习的任务中。传统的奖励机制只能为整个回答提供单一的二元评分，对于具有多个子部分的复杂问题来说过于粗放，难以引导模型通过这些问题。为此，我们提出了StructVRM方法，该方法将多模态推理与结构化可验证奖励模型对齐。其核心是一个基于模型的验证器，该验证器被训练为提供精细的、按子问题级的反馈，评估语义和数学等价性，而不是依赖于严格的字符串匹配。这使得在先前难以处理的问题格式中实现了细微的部分正确评分。广泛的实验展示了StructVRM的有效性。我们的训练模型Seed-StructVRM在六个出十二个公开的多模态基准测试和我们新整理的高难度STEM-Bench上取得了最先进的性能。StructVRM的成功验证了使用结构化可验证奖励进行训练是提高多模态模型在复杂现实世界推理领域能力的有效方法。 

---
# Explaining Similarity in Vision-Language Encoders with Weighted Banzhaf Interactions 

**Title (ZH)**: 基于加权瓦伦哈夫相互作用解释视觉-语言编码器中的相似性 

**Authors**: Hubert Baniecki, Maximilian Muschalik, Fabian Fumagalli, Barbara Hammer, Eyke Hüllermeier, Przemyslaw Biecek  

**Link**: [PDF](https://arxiv.org/pdf/2508.05430)  

**Abstract**: Language-image pre-training (LIP) enables the development of vision-language models capable of zero-shot classification, localization, multimodal retrieval, and semantic understanding. Various explanation methods have been proposed to visualize the importance of input image-text pairs on the model's similarity outputs. However, popular saliency maps are limited by capturing only first-order attributions, overlooking the complex cross-modal interactions intrinsic to such encoders. We introduce faithful interaction explanations of LIP models (FIxLIP) as a unified approach to decomposing the similarity in vision-language encoders. FIxLIP is rooted in game theory, where we analyze how using the weighted Banzhaf interaction index offers greater flexibility and improves computational efficiency over the Shapley interaction quantification framework. From a practical perspective, we propose how to naturally extend explanation evaluation metrics, like the pointing game and area between the insertion/deletion curves, to second-order interaction explanations. Experiments on MS COCO and ImageNet-1k benchmarks validate that second-order methods like FIxLIP outperform first-order attribution methods. Beyond delivering high-quality explanations, we demonstrate the utility of FIxLIP in comparing different models like CLIP vs. SigLIP-2 and ViT-B/32 vs. ViT-L/16. 

**Abstract (ZH)**: LIP模型的忠实交互解释：从一阶到二阶交互解释 

---
# Multi-Modal Multi-Behavior Sequential Recommendation with Conditional Diffusion-Based Feature Denoising 

**Title (ZH)**: 基于条件扩散特征去噪的多模态多行为序列推荐 

**Authors**: Xiaoxi Cui, Weihai Lu, Yu Tong, Yiheng Li, Zhejun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2508.05352)  

**Abstract**: The sequential recommendation system utilizes historical user interactions to predict preferences. Effectively integrating diverse user behavior patterns with rich multimodal information of items to enhance the accuracy of sequential recommendations is an emerging and challenging research direction. This paper focuses on the problem of multi-modal multi-behavior sequential recommendation, aiming to address the following challenges: (1) the lack of effective characterization of modal preferences across different behaviors, as user attention to different item modalities varies depending on the behavior; (2) the difficulty of effectively mitigating implicit noise in user behavior, such as unintended actions like accidental clicks; (3) the inability to handle modality noise in multi-modal representations, which further impacts the accurate modeling of user preferences. To tackle these issues, we propose a novel Multi-Modal Multi-Behavior Sequential Recommendation model (M$^3$BSR). This model first removes noise in multi-modal representations using a Conditional Diffusion Modality Denoising Layer. Subsequently, it utilizes deep behavioral information to guide the denoising of shallow behavioral data, thereby alleviating the impact of noise in implicit feedback through Conditional Diffusion Behavior Denoising. Finally, by introducing a Multi-Expert Interest Extraction Layer, M$^3$BSR explicitly models the common and specific interests across behaviors and modalities to enhance recommendation performance. Experimental results indicate that M$^3$BSR significantly outperforms existing state-of-the-art methods on benchmark datasets. 

**Abstract (ZH)**: 多-line 多态 夙行为 顺序推荐系统（M$^3$$$$- $BSR）通过条件扩散去噪层去除多模态表示表表示表征中的噪声，，随后利用深层行为对抗浅层行为中的的噪声，，最终通过引入多--专Expert 兴趣提取层，在$3$1$BSR）显式建模行为和模态上的的兴趣以提升推荐性能。实验表明，$$d$3$BSR）在基准数据集上显著优于现有最先进的的方法。 

---
# RegionMed-CLIP: A Region-Aware Multimodal Contrastive Learning Pre-trained Model for Medical Image Understanding 

**Title (ZH)**: RegionMed-CLIP: 一种区域意识的多模态对比学习预训练模型用于医学图像理解 

**Authors**: Tianchen Fang, Guiru Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.05244)  

**Abstract**: Medical image understanding plays a crucial role in enabling automated diagnosis and data-driven clinical decision support. However, its progress is impeded by two primary challenges: the limited availability of high-quality annotated medical data and an overreliance on global image features, which often miss subtle but clinically significant pathological regions. To address these issues, we introduce RegionMed-CLIP, a region-aware multimodal contrastive learning framework that explicitly incorporates localized pathological signals along with holistic semantic representations. The core of our method is an innovative region-of-interest (ROI) processor that adaptively integrates fine-grained regional features with the global context, supported by a progressive training strategy that enhances hierarchical multimodal alignment. To enable large-scale region-level representation learning, we construct MedRegion-500k, a comprehensive medical image-text corpus that features extensive regional annotations and multilevel clinical descriptions. Extensive experiments on image-text retrieval, zero-shot classification, and visual question answering tasks demonstrate that RegionMed-CLIP consistently exceeds state-of-the-art vision language models by a wide margin. Our results highlight the critical importance of region-aware contrastive pre-training and position RegionMed-CLIP as a robust foundation for advancing multimodal medical image understanding. 

**Abstract (ZH)**: Medical图像理解在实现自动化诊断和数据驱动的临床决策支持中起着关键作用，但由于高质量标注医疗数据的有限可用性和过度依赖全局图像特征导致的挑战而受阻。为解决这些问题，我们提出了RegionMed-CLIP，这是一种区域意识的多模态对比学习框架，明确结合了局部病理信号和整体语义表示。我们的方法的核心是一种创新的感兴趣区域（ROI）处理器，该处理器适应性地将细粒度的区域特征与全局上下文相结合，并通过逐步训练策略增强层次多模态对齐。为实现大规模区域级表示学习，我们构建了MedRegion-500k，这是一个包含广泛区域注释和多层次临床描述的综合医疗图像-文本语料库。在图像-文本检索、零样本分类和视觉问答任务上的广泛实验表明，RegionMed-CLIP在视觉语言模型上表现出了显著的优势。我们的结果突显了区域意识对比预训练的关键重要性，并将RegionMed-CLIP定位为推动多模态医学图像理解的基础。 

---
# Revealing Temporal Label Noise in Multimodal Hateful Video Classification 

**Title (ZH)**: 揭示多模态仇恨视频分类中的时序标签噪声 

**Authors**: Shuonan Yang, Tailin Chen, Rahul Singh, Jiangbei Yue, Jianbo Jiao, Zeyu Fu  

**Link**: [PDF](https://arxiv.org/pdf/2508.04900)  

**Abstract**: The rapid proliferation of online multimedia content has intensified the spread of hate speech, presenting critical societal and regulatory challenges. While recent work has advanced multimodal hateful video detection, most approaches rely on coarse, video-level annotations that overlook the temporal granularity of hateful content. This introduces substantial label noise, as videos annotated as hateful often contain long non-hateful segments. In this paper, we investigate the impact of such label ambiguity through a fine-grained approach. Specifically, we trim hateful videos from the HateMM and MultiHateClip English datasets using annotated timestamps to isolate explicitly hateful segments. We then conduct an exploratory analysis of these trimmed segments to examine the distribution and characteristics of both hateful and non-hateful content. This analysis highlights the degree of semantic overlap and the confusion introduced by coarse, video-level annotations. Finally, controlled experiments demonstrated that time-stamp noise fundamentally alters model decision boundaries and weakens classification confidence, highlighting the inherent context dependency and temporal continuity of hate speech expression. Our findings provide new insights into the temporal dynamics of multimodal hateful videos and highlight the need for temporally aware models and benchmarks for improved robustness and interpretability. Code and data are available at this https URL. 

**Abstract (ZH)**: 在线多媒体内容的快速 proliferate 加剧了仇恨言论的传播，提出了重要的社会和监管挑战。尽管最近的工作推进了多模态仇恨视频检测的发展，但大多数方法依赖于粗略的视频级别注释，忽视了仇恨内容的时间粒度。这引入了大量标签噪声，因为被标注为仇恨的视频中往往包含长时间的非仇恨片段。在本文中，我们通过细粒度的方法研究了这种标签含糊性的影响。具体来说，我们使用标注的时间戳从 HateMM 和 MultiHateClip 英文数据集中裁剪出仇恨视频，以隔离明确的仇恨片段。然后，我们对这些裁剪片段进行探索性分析，以检查仇恨和非仇恨内容的分布和特征。此分析突显了粗略视频级别注释所引入的语义重叠和混淆。最后，受控实验表明时间戳噪声根本上改变了模型的决策边界并削弱了分类信心，突显了仇恨言论表达的内在上下文依赖性和时间连续性。我们的研究结果提供了有关多模态仇恨视频时间动态的新见解，并强调了对于提高鲁棒性和可解释性的需要使用时间意识模型和基准。代码和数据可在以下网址获取。 

---
