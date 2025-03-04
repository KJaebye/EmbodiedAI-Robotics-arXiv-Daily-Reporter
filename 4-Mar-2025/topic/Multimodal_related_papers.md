# KeyFace: Expressive Audio-Driven Facial Animation for Long Sequences via KeyFrame Interpolation 

**Title (ZH)**: KeyFace：通过关键帧插值实现长序列的表达性音频驱动面部动画 

**Authors**: Antoni Bigata, Michał Stypułkowski, Rodrigo Mira, Stella Bounareli, Konstantinos Vougioukas, Zoe Landgraf, Nikita Drobyshev, Maciej Zieba, Stavros Petridis, Maja Pantic  

**Link**: [PDF](https://arxiv.org/pdf/2503.01715)  

**Abstract**: Current audio-driven facial animation methods achieve impressive results for short videos but suffer from error accumulation and identity drift when extended to longer durations. Existing methods attempt to mitigate this through external spatial control, increasing long-term consistency but compromising the naturalness of motion. We propose KeyFace, a novel two-stage diffusion-based framework, to address these issues. In the first stage, keyframes are generated at a low frame rate, conditioned on audio input and an identity frame, to capture essential facial expressions and movements over extended periods of time. In the second stage, an interpolation model fills in the gaps between keyframes, ensuring smooth transitions and temporal coherence. To further enhance realism, we incorporate continuous emotion representations and handle a wide range of non-speech vocalizations (NSVs), such as laughter and sighs. We also introduce two new evaluation metrics for assessing lip synchronization and NSV generation. Experimental results show that KeyFace outperforms state-of-the-art methods in generating natural, coherent facial animations over extended durations, successfully encompassing NSVs and continuous emotions. 

**Abstract (ZH)**: 基于关键帧的双重扩散框架KeyFace：实现长时间连续发音的情感面部动画 

---
# MINT: Multi-modal Chain of Thought in Unified Generative Models for Enhanced Image Generation 

**Title (ZH)**: MINT: 统一生成模型中多模态思维链在增强图像生成中的应用 

**Authors**: Yi Wang, Mushui Liu, Wanggui He, Longxiang Zhang, Ziwei Huang, Guanghao Zhang, Fangxun Shu, Zhong Tao, Dong She, Zhelun Yu, Haoyuan Li, Weilong Dai, Mingli Song, Jie Song, Hao Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2503.01298)  

**Abstract**: Unified generative models have demonstrated extraordinary performance in both text and image generation. However, they tend to underperform when generating intricate images with various interwoven conditions, which is hard to solely rely on straightforward text-to-image generation. In response to this challenge, we introduce MINT, an innovative unified generative model, empowered with native multimodal chain of thought (MCoT) for enhanced image generation for the first time. Firstly, we design Mixture of Transformer Experts (MTXpert), an expert-parallel structure that effectively supports both natural language generation (NLG) and visual capabilities, while avoiding potential modality conflicts that could hinder the full potential of each modality. Building on this, we propose an innovative MCoT training paradigm, a step-by-step approach to multimodal thinking, reasoning, and reflection specifically designed to enhance image generation. This paradigm equips MINT with nuanced, element-wise decoupled alignment and a comprehensive understanding of textual and visual components. Furthermore, it fosters advanced multimodal reasoning and self-reflection, enabling the construction of images that are firmly grounded in the logical relationships between these elements. Notably, MINT has been validated to exhibit superior performance across multiple benchmarks for text-to-image (T2I) and image-to-text (I2T) tasks. 

**Abstract (ZH)**: 统一生成模型在文本和图像生成任务中表现出色，但在生成具有多种交织条件的复杂图像时往往表现不佳，单纯依赖文本到图像生成难以解决这一问题。为应对这一挑战，我们引入了一种名为MINT的创新统一生成模型，该模型集成了原生多模态链式思维（MCoT），首次提升了图像生成能力。首先，我们设计了混合Transformer专家（MTXpert）结构，有效支持自然语言生成和视觉能力，同时避免了可能阻碍每种模态潜力的模态冲突。在此基础上，我们提出了一种创新的MCoT训练范式，这是一种逐步的多模态思维、推理和反思过程，旨在增强图像生成能力。该范式为MINT提供了细腻的元素级解耦对齐和对文本和视觉成分的全面理解。此外，它促进了高级的多模态推理和自我反思，使MINT能够构建逻辑关系紧密的图像。值得注意的是，MINT在多种文本到图像（T2I）和图像到文本（I2T）任务基准测试中表现优越。 

---
# MedUnifier: Unifying Vision-and-Language Pre-training on Medical Data with Vision Generation Task using Discrete Visual Representations 

**Title (ZH)**: MedUnifier: 使用离散视觉表示在医学数据上统一视觉和语言预训练以及视觉生成任务 

**Authors**: Ziyang Zhang, Yang Yu, Yucheng Chen, Xulei Yang, Si Yong Yeo  

**Link**: [PDF](https://arxiv.org/pdf/2503.01019)  

**Abstract**: Despite significant progress in Vision-Language Pre-training (VLP), current approaches predominantly emphasize feature extraction and cross-modal comprehension, with limited attention to generating or transforming visual content. This gap hinders the model's ability to synthesize coherent and novel visual representations from textual prompts, thereby reducing the effectiveness of multi-modal learning. In this work, we propose MedUnifier, a unified VLP framework tailored for medical data. MedUnifier seamlessly integrates text-grounded image generation capabilities with multi-modal learning strategies, including image-text contrastive alignment, image-text matching and image-grounded text generation. Unlike traditional methods that reply on continuous visual representations, our approach employs visual vector quantization, which not only facilitates a more cohesive learning strategy for cross-modal understanding but also enhances multi-modal generation quality by effectively leveraging discrete representations. Our framework's effectiveness is evidenced by the experiments on established benchmarks, including uni-modal tasks (supervised fine-tuning), cross-modal tasks (image-text retrieval and zero-shot image classification), and multi-modal tasks (medical report generation, image synthesis), where it achieves state-of-the-art performance across various tasks. MedUnifier also offers a highly adaptable tool for a wide range of language and vision tasks in healthcare, marking advancement toward the development of a generalizable AI model for medical applications. 

**Abstract (ZH)**: MedUnifier：一种面向医疗数据的统一视觉语言预训练框架 

---
# Cross Modality Medical Image Synthesis for Improving Liver Segmentation 

**Title (ZH)**: 跨模态医学图像合成以改善肝脏分割 

**Authors**: Muhammad Rafiq, Hazrat Ali, Ghulam Mujtaba, Zubair Shah, Shoaib Azmat  

**Link**: [PDF](https://arxiv.org/pdf/2503.00945)  

**Abstract**: Deep learning-based computer-aided diagnosis (CAD) of medical images requires large datasets. However, the lack of large publicly available labeled datasets limits the development of deep learning-based CAD systems. Generative Adversarial Networks (GANs), in particular, CycleGAN, can be used to generate new cross-domain images without paired training data. However, most CycleGAN-based synthesis methods lack the potential to overcome alignment and asymmetry between the input and generated data. We propose a two-stage technique for the synthesis of abdominal MRI using cross-modality translation of abdominal CT. We show that the synthetic data can help improve the performance of the liver segmentation network. We increase the number of abdominal MRI images through cross-modality image transformation of unpaired CT images using a CycleGAN inspired deformation invariant network called EssNet. Subsequently, we combine the synthetic MRI images with the original MRI images and use them to improve the accuracy of the U-Net on a liver segmentation task. We train the U-Net on real MRI images and then on real and synthetic MRI images. Consequently, by comparing both scenarios, we achieve an improvement in the performance of U-Net. In summary, the improvement achieved in the Intersection over Union (IoU) is 1.17%. The results show potential to address the data scarcity challenge in medical imaging. 

**Abstract (ZH)**: 基于深度学习的医学图像计算机辅助诊断（CAD）需要大量数据。然而，缺乏大型公开标注数据集限制了基于深度学习的CAD系统的开发。生成对抗网络（GAN），特别是CycleGAN，可以在没有配对训练数据的情况下生成新的跨域图像。然而，大多数基于CycleGAN的合成方法在解决输入数据和生成数据之间的对齐和不对称性方面缺乏潜力。我们提出了一种两阶段技术，通过腹部CT的跨模态转换生成腹部MRI图像。我们展示了合成数据可以帮助提高肝脏分割网络的性能。我们通过使用一种称为EssNet的变形不变网络，将未配对的CT图像进行跨模态图像变换，增加腹部MRI图像的数量。随后，我们将合成的MRI图像与原始MRI图像结合，并利用它们改善U-Net在肝脏分割任务中的准确性。我们先在真实的MRI图像上训练U-Net，然后在真实的和合成的MRI图像上训练。最终，通过比较两种场景，我们实现了U-Net性能的提升。概括而言，我们在Intersection over Union（IoU）上的改进达到了1.17%。结果展示了解决医学成像中的数据稀缺挑战的潜力。 

---
# PinLanding: Content-First Keyword Landing Page Generation via Multi-Modal AI for Web-Scale Discovery 

**Title (ZH)**: PinLanding：基于多模态AI的内容优先关键字着陆页生成以实现大规模网络发现 

**Authors**: Faye Zhang, Jasmine Wan, Qianyu Cheng, Jinfeng Rao  

**Link**: [PDF](https://arxiv.org/pdf/2503.00619)  

**Abstract**: Online platforms like Pinterest hosting vast content collections traditionally rely on manual curation or user-generated search logs to create keyword landing pages (KLPs) -- topic-centered collection pages that serve as entry points for content discovery. While manual curation ensures quality, it doesn't scale to millions of collections, and search log approaches result in limited topic coverage and imprecise content matching. In this paper, we present PinLanding, a novel content-first architecture that transforms the way platforms create topical collections. Instead of deriving topics from user behavior, our system employs a multi-stage pipeline combining vision-language model (VLM) for attribute extraction, large language model (LLM) for topic generation, and a CLIP-based dual-encoder architecture for precise content matching. Our model achieves 99.7% Recall@10 on Fashion200K benchmark, demonstrating strong attribute understanding capabilities. In production deployment for search engine optimization with 4.2 million shopping landing pages, the system achieves a 4X increase in topic coverage and 14.29% improvement in collection attribute precision over the traditional search log-based approach via human evaluation. The architecture can be generalized beyond search traffic to power various user experiences, including content discovery and recommendations, providing a scalable solution to transform unstructured content into curated topical collections across any content domain. 

**Abstract (ZH)**: 基于内容的第一代PinLanding：一种新型的话题集合生成架构 

---
# SafeAuto: Knowledge-Enhanced Safe Autonomous Driving with Multimodal Foundation Models 

**Title (ZH)**: SafeAuto：多模态基础模型增强的安全自动驾驶 

**Authors**: Jiawei Zhang, Xuan Yang, Taiqi Wang, Yu Yao, Aleksandr Petiushko, Bo Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.00211)  

**Abstract**: Traditional autonomous driving systems often struggle to integrate high-level reasoning with low-level control, resulting in suboptimal and sometimes unsafe driving behaviors. The emergence of Multimodal Large Language Models (MLLMs), which can process both visual and textual data, presents an opportunity to unify perception and reasoning tasks within a single framework. However, effectively embedding precise safety knowledge into MLLMs for autonomous driving remains a significant challenge. To address this, we propose SafeAuto, a novel framework that enhances MLLM-based autonomous driving systems by incorporating both unstructured and structured knowledge. Specifically, we first introduce the Position-Dependent Cross-Entropy (PDCE) loss function, designed to improve the accuracy of low-level control signal predictions when numerical values are represented as text. Second, to ensure safe autonomous driving by explicitly integrating precise safety knowledge into the MLLM, we develop a reasoning component for SafeAuto. This component translates driving safety regulations into first-order logic rules (e.g., "red light => stop") and incorporates these rules into a probabilistic graphical model, such as a Markov Logic Network (MLN). The MLN is trained to verify the predicted next actions using environmental attributes identified by attribute recognition models (e.g., detecting a red light) to form the predicates. Additionally, we construct a Multimodal RAG model that leverages video data, control signals, and environmental attributes to learn more effectively from past similar driving experiences. By integrating PDCE, MLN, and Multimodal RAG, SafeAuto significantly outperforms existing baselines across multiple datasets. This advancement enables more accurate, reliable, and safer autonomous driving systems that learn from experience, obey traffic laws, and perform precise control actions. 

**Abstract (ZH)**: 传统自动驾驶系统往往难以将高层次推理与低层次控制相结合，导致驾驶行为不够优化甚至存在安全隐患。多模式大语言模型（MLLMs）能够处理视觉和文本数据，为统一感知和推理任务提供机会。然而，将精确的安全知识有效嵌入到MLLMs以实现自动驾驶仍然面临重大挑战。为此，我们提出了SafeAuto框架，通过结合非结构化和结构化知识来增强基于MLLM的自动驾驶系统。具体而言，我们首先引入位置相关交叉熵（PDCE）损失函数，旨在提高数值表示为文本的低层次控制信号预测准确性。其次，为了通过显式集成精确的安全知识确保安全的自动驾驶，我们为SafeAuto开发了一个推理组件。该组件将驾驶安全规定翻译成一阶逻辑规则（例如，“红灯=>停止”），并将这些规则整合到概率图形模型中，如马尔可夫逻辑网络（MLN）。MLN将通过环境属性识别模型（如检测红灯）识别的环境属性转换为谓词，并对其进行训练以验证预测的下一步行动。此外，我们构建了一个多模式RAG模型，利用视频数据、控制信号和环境属性，更有效地从过去的类似驾驶经验中学习。通过结合PDCE、MLN和多模式RAG，SafeAuto在多个数据集中显著优于现有基线。这一进展使得自动驾驶系统能够更准确、可靠且安全地学习，遵守交通法规并执行精确的控制操作。 

---
# PaliGemma-CXR: A Multi-task Multimodal Model for TB Chest X-ray Interpretation 

**Title (ZH)**: PaliGemma-CXR：一种用于TB胸部X光片解释的多任务多模态模型 

**Authors**: Denis Musinguzi, Andrew Katumba, Sudi Murindanyi  

**Link**: [PDF](https://arxiv.org/pdf/2503.00171)  

**Abstract**: Tuberculosis (TB) is a infectious global health challenge. Chest X-rays are a standard method for TB screening, yet many countries face a critical shortage of radiologists capable of interpreting these images. Machine learning offers an alternative, as it can automate tasks such as disease diagnosis, and report generation. However, traditional approaches rely on task-specific models, which cannot utilize the interdependence between tasks. Building a multi-task model capable of performing multiple tasks poses additional challenges such as scarcity of multimodal data, dataset imbalance, and negative transfer. To address these challenges, we propose PaliGemma-CXR, a multi-task multimodal model capable of performing TB diagnosis, object detection, segmentation, report generation, and VQA. Starting with a dataset of chest X-ray images annotated with TB diagnosis labels and segmentation masks, we curated a multimodal dataset to support additional tasks. By finetuning PaliGemma on this dataset and sampling data using ratios of the inverse of the size of task datasets, we achieved the following results across all tasks: 90.32% accuracy on TB diagnosis and 98.95% on close-ended VQA, 41.3 BLEU score on report generation, and a mAP of 19.4 and 16.0 on object detection and segmentation, respectively. These results demonstrate that PaliGemma-CXR effectively leverages the interdependence between multiple image interpretation tasks to enhance performance. 

**Abstract (ZH)**: tuberculosis (tb) 是一个全球性的健康挑战。胸部x射线是tb筛查的标准方法，但许多国家面临放射科医生严重短缺的问题，难以解读这些影像。机器学习提供了一种替代方案，它可以自动化疾病诊断和报告生成等任务。然而，传统方法依赖于特定任务的模型，无法利用任务间的相互依赖性。构建能够执行多个任务的多任务模型带来了额外的挑战，如多模态数据稀缺、数据集不平衡和负迁移等问题。为了解决这些问题，我们提出了PaliGemma-CXR，这是一种能够执行tb诊断、对象检测、分割、报告生成和VQA的多任务多模态模型。我们从带有tb诊断标签和分割掩码的胸部x射线图像数据集开始，构建了一个多模态数据集以支持额外的任务。通过对PaliGemma进行微调并在任务数据集大小的倒数比例下采样数据，我们在所有任务上取得了以下结果：tb诊断的90.32%准确率和闭合式vqa的98.95%准确率、报告生成的41.3 bleu分数、对象检测和分割的mAP分别为19.4和16.0。这些结果表明，PaliGemma-CXR有效地利用了多种图像解释任务之间的相互依赖性以提高性能。 

---
# PreMind: Multi-Agent Video Understanding for Advanced Indexing of Presentation-style Videos 

**Title (ZH)**: 预心智：多智能体视频理解及其在演讲风格视频高级索引中的应用 

**Authors**: Kangda Wei, Zhengyu Zhou, Bingqing Wang, Jun Araki, Lukas Lange, Ruihong Huang, Zhe Feng  

**Link**: [PDF](https://arxiv.org/pdf/2503.00162)  

**Abstract**: In recent years, online lecture videos have become an increasingly popular resource for acquiring new knowledge. Systems capable of effectively understanding/indexing lecture videos are thus highly desirable, enabling downstream tasks like question answering to help users efficiently locate specific information within videos. This work proposes PreMind, a novel multi-agent multimodal framework that leverages various large models for advanced understanding/indexing of presentation-style videos. PreMind first segments videos into slide-presentation segments using a Vision-Language Model (VLM) to enhance modern shot-detection techniques. Each segment is then analyzed to generate multimodal indexes through three key steps: (1) extracting slide visual content, (2) transcribing speech narratives, and (3) consolidating these visual and speech contents into an integrated understanding. Three innovative mechanisms are also proposed to improve performance: leveraging prior lecture knowledge to refine visual understanding, detecting/correcting speech transcription errors using a VLM, and utilizing a critic agent for dynamic iterative self-reflection in vision analysis. Compared to traditional video indexing methods, PreMind captures rich, reliable multimodal information, allowing users to search for details like abbreviations shown only on slides. Systematic evaluations on the public LPM dataset and an internal enterprise dataset are conducted to validate PreMind's effectiveness, supported by detailed analyses. 

**Abstract (ZH)**: 近年来，线上讲座视频已成为获取新知识越来越重要的资源。能够有效理解/索引讲座视频的系统因此变得尤为重要，这些系统可以支持下游任务如问答，帮助用户高效地定位视频中的特定信息。本文提出PreMind，这是一种新颖的多代理多模态框架，利用多种大规模模型对演示型视频进行高级的理解/索引。PreMind首先使用视觉-语言模型（VLM）将视频分割成幻灯片演示段，以增强现代镜头检测技术。每个段落然后通过三个关键步骤进行分析以生成多模态索引：（1）提取幻灯片视觉内容，（2）转述演讲叙述，（3）将这些视觉和语音内容整合为统一的理解。还提出了三种创新机制以提高性能：利用先验讲座知识 refining 视觉理解，使用视觉语言模型检测/校正语音转录错误，以及利用批评代理进行动态迭代自我反思在视觉分析中的应用。与传统的视频索引方法相比，PreMind能够捕获丰富可靠的多模态信息，使用户能够搜索仅在幻灯片上显示的缩写等细节。在公共LPM数据集和内部企业数据集上的系统评估证实了PreMind的有效性，通过详细的分析支持。 

---
# Deciphering the complaint aspects: Towards an aspect-based complaint identification model with video complaint dataset in finance 

**Title (ZH)**: 解读投诉方面：面向金融领域基于视频投诉数据的方面导向投诉识别模型研究 

**Authors**: Sarmistha Das, Basha Mujavarsheik, R E Zera Lyngkhoi, Sriparna Saha, Alka Maurya  

**Link**: [PDF](https://arxiv.org/pdf/2503.00054)  

**Abstract**: In today's competitive marketing landscape, effective complaint management is crucial for customer service and business success. Video complaints, integrating text and image content, offer invaluable insights by addressing customer grievances and delineating product benefits and drawbacks. However, comprehending nuanced complaint aspects within vast daily multimodal financial data remains a formidable challenge. Addressing this gap, we have curated a proprietary multimodal video complaint dataset comprising 433 publicly accessible instances. Each instance is meticulously annotated at the utterance level, encompassing five distinct categories of financial aspects and their associated complaint labels. To support this endeavour, we introduce Solution 3.0, a model designed for multimodal aspect-based complaint identification task. Solution 3.0 is tailored to perform three key tasks: 1) handling multimodal features ( audio and video), 2) facilitating multilabel aspect classification, and 3) conducting multitasking for aspect classifications and complaint identification parallelly. Solution 3.0 utilizes a CLIP-based dual frozen encoder with an integrated image segment encoder for global feature fusion, enhanced by contextual attention (ISEC) to improve accuracy and efficiency. Our proposed framework surpasses current multimodal baselines, exhibiting superior performance across nearly all metrics by opening new ways to strengthen appropriate customer care initiatives and effectively assisting individuals in resolving their problems. 

**Abstract (ZH)**: 在当今竞争激烈的营销环境中，有效的投诉管理对于客户服务和商业成功至关重要。结合文本和图像内容的视频投诉提供了宝贵的见解，有助于解决客户问题并阐述产品的优缺点。然而，在大量日常多模态金融数据中理解细微的投诉方面仍然是一项艰巨的挑战。为应对这一挑战，我们精心构建了一个包含433个公开实例的多模态视频投诉数据集。每个实例在语句级别详细标注了五个不同类别的金融方面及其相应的投诉标签。为此，我们引入了Solution 3.0模型，该模型旨在完成三项关键任务：1）处理多模态特征（音频和视频），2）促进多标签方面分类，3）并行进行方面分类和投诉识别的多任务处理。Solution 3.0利用基于CLIP的双冻结编码器结合图像段编码器进行全局特征融合，并通过上下文注意力（ISEC）提高准确性和效率。我们提出的框架超越了现有的多模态基线，几乎在所有指标上都表现出色，为加强适当客户关怀举措并有效帮助个人解决其问题开辟了新途径。 

---
# Zero-Shot Defense Against Toxic Images via Inherent Multimodal Alignment in LVLMs 

**Title (ZH)**: 零样本防御针对有毒图像的内在多模态对齐在LVLMs中 

**Authors**: Wei Zhao, Zhe Li, Yige Li, Jun Sun  

**Link**: [PDF](https://arxiv.org/pdf/2503.00037)  

**Abstract**: Large Vision-Language Models (LVLMs) have made significant strides in multimodal comprehension, thanks to extensive pre-training and fine-tuning on large-scale visual datasets. However, despite their robust textual safety mechanisms, they remain vulnerable to harmful visual inputs. Existing safeguards-typically relying on pre-filtering or fine-tuning-incur high costs and diminish overall utility. To address this critical vulnerability, we introduce SafeCLIP, a lightweight method that leverages LVLMs inherent multimodal alignment for zero-shot toxic image detection. By projecting CLIPs discarded CLS token into its text space and matching it with toxic descriptors, SafeCLIP detects harmful content without any architectural changes-adding minimal latency and enabling dynamic safety corrections during inference and this http URL show that SafeCLIP achieves a 66.9% defense success rate with only 3.2% false positive rate and 7.2% overhead. In contrast, state-of-the-art methods achieve 52.9% success but have a 10.7% false positive rate and 210% overhead. Our work demonstrates that leveraging inherent multimodal alignment can yield efficient, low-cost LVLM safety. Code is available at this http URL. 

**Abstract (ZH)**: 大型多模态语言视觉模型（LVLMs）在多模态理解方面取得了显著进展，得益于其在大规模视觉数据集上的广泛预训练和微调。然而，尽管它们具有稳健的文本安全性机制，仍然容易受到有害视觉输入的影响。现有的防护措施通常依赖于预过滤或微调，会导致高成本并降低整体实用性。为此，我们引入了SafeCLIP，一种轻量级的方法，利用LVLMs固有的多模态对齐进行零样本有毒图像检测。SafeCLIP通过将CLIP丢弃的CLS-token投影到文本空间并与有毒描述符匹配来检测有害内容，无需架构更改，增加的延迟最少，并能够在推理期间实现动态安全性修正。实验结果表明，SafeCLIP达到了66.9%的防御成功率，仅产生3.2%的误报率和7.2%的开销。相比之下，最先进的方法成功率仅为52.9%，误报率为10.7%，开销为210%。我们的研究证明，利用固有的多模态对齐可以实现高效且低成本的LVLM安全性。相关代码可在<此链接>获取。 

---
