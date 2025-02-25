# NatSGLD: A Dataset with Speech, Gesture, Logic, and Demonstration for Robot Learning in Natural Human-Robot Interaction 

**Title (ZH)**: NatSGLD：一种用于自然人机交互中机器人学习的数据集，包含语音、手势、逻辑和演示内容 

**Authors**: Snehesh Shrestha, Yantian Zha, Saketh Banagiri, Ge Gao, Yiannis Aloimonos, Cornelia Fermüller  

**Link**: [PDF](https://arxiv.org/pdf/2502.16718)  

**Abstract**: Recent advances in multimodal Human-Robot Interaction (HRI) datasets emphasize the integration of speech and gestures, allowing robots to absorb explicit knowledge and tacit understanding. However, existing datasets primarily focus on elementary tasks like object pointing and pushing, limiting their applicability to complex domains. They prioritize simpler human command data but place less emphasis on training robots to correctly interpret tasks and respond appropriately. To address these gaps, we present the NatSGLD dataset, which was collected using a Wizard of Oz (WoZ) method, where participants interacted with a robot they believed to be autonomous. NatSGLD records humans' multimodal commands (speech and gestures), each paired with a demonstration trajectory and a Linear Temporal Logic (LTL) formula that provides a ground-truth interpretation of the commanded tasks. This dataset serves as a foundational resource for research at the intersection of HRI and machine learning. By providing multimodal inputs and detailed annotations, NatSGLD enables exploration in areas such as multimodal instruction following, plan recognition, and human-advisable reinforcement learning from demonstrations. We release the dataset and code under the MIT License at this https URL to support future HRI research. 

**Abstract (ZH)**: Recent Advances in Multimodal Human-Robot Interaction (HRI) Datasets Emphasize the Integration of Speech and Gestures: The NatSGLD Dataset Fills Gaps in Complex Task Understanding 

---
# Multimodal Inconsistency Reasoning (MMIR): A New Benchmark for Multimodal Reasoning Models 

**Title (ZH)**: 多模态不一致性推理（MMIR）：多模态推理模型的新基准 

**Authors**: Qianqi Yan, Yue Fan, Hongquan Li, Shan Jiang, Yang Zhao, Xinze Guan, Ching-Chen Kuo, Xin Eric Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.16033)  

**Abstract**: Existing Multimodal Large Language Models (MLLMs) are predominantly trained and tested on consistent visual-textual inputs, leaving open the question of whether they can handle inconsistencies in real-world, layout-rich content. To bridge this gap, we propose the Multimodal Inconsistency Reasoning (MMIR) benchmark to assess MLLMs' ability to detect and reason about semantic mismatches in artifacts such as webpages, presentation slides, and posters. MMIR comprises 534 challenging samples, each containing synthetically injected errors across five reasoning-heavy categories: Factual Contradiction, Identity Misattribution, Contextual Mismatch, Quantitative Discrepancy, and Temporal/Spatial Incoherence. We evaluate six state-of-the-art MLLMs, showing that models with dedicated multimodal reasoning capabilities, such as o1, substantially outperform their counterparts while open-source models remain particularly vulnerable to inconsistency errors. Detailed error analyses further show that models excel in detecting inconsistencies confined to a single modality, particularly in text, but struggle with cross-modal conflicts and complex layouts. Probing experiments reveal that single-modality prompting, including Chain-of-Thought (CoT) and Set-of-Mark (SoM) methods, yields marginal gains, revealing a key bottleneck in cross-modal reasoning. Our findings highlight the need for advanced multimodal reasoning and point to future research on multimodal inconsistency. 

**Abstract (ZH)**: 现有多种模态大型语言模型（MLLMs）主要在一致的视觉-文本输入上进行训练和测试，这留下了一个问题，即它们是否能够处理现实世界中布局丰富的不一致性内容。为了弥合这一差距，我们提出了多模态不一致性推理（MMIR）基准，以评估MLLMs在检测和推理艺术品（如网页、演示文稿和海报中的语义不匹配）方面的能力。MMIR包含534个具有挑战性的样本，每个样本都含有在五个推理密集类别中注入的合成错误：事实矛盾、身份误归因、上下文不匹配、定量偏差和时空不连贯性。我们评估了六种最先进的MLLMs，结果显示，具有专门多模态推理能力的模型，如o1，显著优于其同类模型，而开源模型则特别容易受到不一致性错误的影响。详细的错误分析进一步表明，模型在检测单一模态内的不一致性方面表现出色，特别是在文本中，但在跨模态冲突和复杂布局方面则表现不佳。探针实验揭示，单一模态提示，包括思维链（CoT）和标记集（SoM）方法，仅带来微弱的收益，显示出跨模态推理的关键瓶颈。我们的研究结果突出了先进的多模态推理的需求，并指出了未来多模态不一致性研究的方向。 

---
# MLLMs Know Where to Look: Training-free Perception of Small Visual Details with Multimodal LLMs 

**Title (ZH)**: MLLMs 知道该注视何处：基于多模态大语言模型的无训练感知小视觉细节 

**Authors**: Jiarui Zhang, Mahyar Khayatkhoei, Prateek Chhikara, Filip Ilievski  

**Link**: [PDF](https://arxiv.org/pdf/2502.17422)  

**Abstract**: Multimodal Large Language Models (MLLMs) have experienced rapid progress in visual recognition tasks in recent years. Given their potential integration into many critical applications, it is important to understand the limitations of their visual perception. In this work, we study whether MLLMs can perceive small visual details as effectively as large ones when answering questions about images. We observe that their performance is very sensitive to the size of the visual subject of the question, and further show that this effect is in fact causal by conducting an intervention study. Next, we study the attention patterns of MLLMs when answering visual questions, and intriguingly find that they consistently know where to look, even when they provide the wrong answer. Based on these findings, we then propose training-free visual intervention methods that leverage the internal knowledge of any MLLM itself, in the form of attention and gradient maps, to enhance its perception of small visual details. We evaluate our proposed methods on two widely-used MLLMs and seven visual question answering benchmarks and show that they can significantly improve MLLMs' accuracy without requiring any training. Our results elucidate the risk of applying MLLMs to visual recognition tasks concerning small details and indicate that visual intervention using the model's internal state is a promising direction to mitigate this risk. 

**Abstract (ZH)**: 多模态大型语言模型在视觉细节识别任务中的局限性及干预方法研究 

---
# Distributional Vision-Language Alignment by Cauchy-Schwarz Divergence 

**Title (ZH)**: 基于柯西-施瓦茨散度的分布视听一致性对齐 

**Authors**: Wenzhe Yin, Zehao Xiao, Pan Zhou, Shujian Yu, Jiayi Shen, Jan-Jakob Sonke, Efstratios Gavves  

**Link**: [PDF](https://arxiv.org/pdf/2502.17028)  

**Abstract**: Multimodal alignment is crucial for various downstream tasks such as cross-modal generation and retrieval. Previous multimodal approaches like CLIP maximize the mutual information mainly by aligning pairwise samples across modalities while overlooking the distributional differences, leading to suboptimal alignment with modality gaps. In this paper, to overcome the limitation, we propose CS-Aligner, a novel and straightforward framework that performs distributional vision-language alignment by integrating Cauchy-Schwarz (CS) divergence with mutual information. In the proposed framework, we find that the CS divergence and mutual information serve complementary roles in multimodal alignment, capturing both the global distribution information of each modality and the pairwise semantic relationships, yielding tighter and more precise alignment. Moreover, CS-Aligher enables incorporating additional information from unpaired data and token-level representations, enhancing flexible and fine-grained alignment in practice. Experiments on text-to-image generation and cross-modality retrieval tasks demonstrate the effectiveness of our method on vision-language alignment. 

**Abstract (ZH)**: 多模态对齐对于跨模态生成和检索等下游任务至关重要。先前的多模态方法如CLIP主要通过在不同模态之间对齐成对样本来最大化互信息，而忽视了模态之间的分布差异，导致了模态差距下的子优对齐。本文为克服这一局限，提出了一种新颖且简单的CS-Aligner框架，该框架通过整合柯西-施瓦茨(CS)散度与互信息来执行分布性的视觉-语言对齐。在提出的框架中，我们发现CS散度和互信息在多模态对齐中发挥互补作用，既捕捉了每个模态的全局分布信息，又捕捉了成对的语义关系，从而实现了更紧密和精确的对齐。此外，CS-Aligner允许从未配对数据和令牌级表示中获取额外信息，增强了实际中的灵活和精细对齐。在文本到图像生成和跨模态检索任务上的实验表明，该方法在视觉-语言对齐方面是有效的。 

---
# Culture-TRIP: Culturally-Aware Text-to-Image Generation with Iterative Prompt Refinment 

**Title (ZH)**: 文化TRIP：具有迭代提示精炼的文化意识文本到图像生成 

**Authors**: Suchae Jeong, Inseong Choi, Youngsik Yun, Jihie Kim  

**Link**: [PDF](https://arxiv.org/pdf/2502.16902)  

**Abstract**: Text-to-Image models, including Stable Diffusion, have significantly improved in generating images that are highly semantically aligned with the given prompts. However, existing models may fail to produce appropriate images for the cultural concepts or objects that are not well known or underrepresented in western cultures, such as `hangari' (Korean utensil). In this paper, we propose a novel approach, Culturally-Aware Text-to-Image Generation with Iterative Prompt Refinement (Culture-TRIP), which refines the prompt in order to improve the alignment of the image with such culture nouns in text-to-image models. Our approach (1) retrieves cultural contexts and visual details related to the culture nouns in the prompt and (2) iteratively refines and evaluates the prompt based on a set of cultural criteria and large language models. The refinement process utilizes the information retrieved from Wikipedia and the Web. Our user survey, conducted with 66 participants from eight different countries demonstrates that our proposed approach enhances the alignment between the images and the prompts. In particular, C-TRIP demonstrates improved alignment between the generated images and underrepresented culture nouns. Resource can be found at this https URL. 

**Abstract (ZH)**: 基于迭代提示精炼的文化意识文本到图像生成（Culture-TRIP） 

---
# MemeIntel: Explainable Detection of Propagandistic and Hateful Memes 

**Title (ZH)**: MemeIntel: 可解释的 propagandistic 和 hateful 病毒贴图检测 

**Authors**: Mohamed Bayan Kmainasi, Abul Hasnat, Md Arid Hasan, Ali Ezzat Shahroor, Firoj Alam  

**Link**: [PDF](https://arxiv.org/pdf/2502.16612)  

**Abstract**: The proliferation of multimodal content on social media presents significant challenges in understanding and moderating complex, context-dependent issues such as misinformation, hate speech, and propaganda. While efforts have been made to develop resources and propose new methods for automatic detection, limited attention has been given to label detection and the generation of explanation-based rationales for predicted labels. To address this challenge, we introduce MemeIntel, an explanation-enhanced dataset for propaganda memes in Arabic and hateful memes in English, making it the first large-scale resource for these tasks. To solve these tasks, we propose a multi-stage optimization approach and train Vision-Language Models (VLMs). Our results demonstrate that this approach significantly improves performance over the base model for both \textbf{label detection} and explanation generation, outperforming the current state-of-the-art with an absolute improvement of ~3% on ArMeme and ~7% on Hateful Memes. For reproducibility and future research, we aim to make the MemeIntel dataset and experimental resources publicly available. 

**Abstract (ZH)**: 社交媒体上多模态内容的泛滥为理解和 moderating 如虚假信息、仇恨言论和宣传等复杂、情境依赖性问题带来了重大挑战。尽管已经做出了努力来开发资源和提出新的自动检测方法，但对标签检测和基于解释的理由生成的关注仍然有限。为应对这一挑战，我们介绍了 MemeIntel，这是一个增强解释的数据集，包含阿拉伯语宣传模因和英语仇恨模因，使其成为这些任务上的首个大规模资源。为了解决这些任务，我们提出了一个多阶段优化方法，并训练了视觉-语言模型（VLMs）。我们的结果表明，这种方法在标签检测和解释生成上显著提升了基线模型的表现，在ArMeme上绝对提升了约3%，在仇恨模因上提升了约7%，超越了当前最先进方法。为了可再现性和未来研究，我们致力于使MemeIntel数据集和实验资源公开可用。 

---
# Cross-domain Few-shot Object Detection with Multi-modal Textual Enrichment 

**Title (ZH)**: 跨领域少样本对象检测的多模态文本丰富方法 

**Authors**: Zeyu Shangguan, Daniel Seita, Mohammad Rostami  

**Link**: [PDF](https://arxiv.org/pdf/2502.16469)  

**Abstract**: Advancements in cross-modal feature extraction and integration have significantly enhanced performance in few-shot learning tasks. However, current multi-modal object detection (MM-OD) methods often experience notable performance degradation when encountering substantial domain shifts. We propose that incorporating rich textual information can enable the model to establish a more robust knowledge relationship between visual instances and their corresponding language descriptions, thereby mitigating the challenges of domain shift. Specifically, we focus on the problem of Cross-Domain Multi-Modal Few-Shot Object Detection (CDMM-FSOD) and introduce a meta-learning-based framework designed to leverage rich textual semantics as an auxiliary modality to achieve effective domain adaptation. Our new architecture incorporates two key components: (i) A multi-modal feature aggregation module, which aligns visual and linguistic feature embeddings to ensure cohesive integration across modalities. (ii) A rich text semantic rectification module, which employs bidirectional text feature generation to refine multi-modal feature alignment, thereby enhancing understanding of language and its application in object detection. We evaluate the proposed method on common cross-domain object detection benchmarks and demonstrate that it significantly surpasses existing few-shot object detection approaches. 

**Abstract (ZH)**: 跨模态特征提取与整合的进展显著提升了少样本学习任务的表现。然而，当前的多模态物体检测（MM-OD）方法在遇到显著领域偏移时常常会表现出明显的性能下降。我们提出通过引入丰富的文本信息，可以使模型建立视觉实例与其对应语言描述之间的更稳健的知识关系，从而缓解领域偏移的挑战。具体而言，我们重点解决跨域多模态少样本物体检测（CDMM-FSOD）问题，并提出一种基于元学习的框架，利用丰富的文本语义作为辅助模态，以实现有效的领域适应。我们的新架构包含两个关键组件：（i）多模态特征聚合模块，该模块对齐视觉和语言特征嵌入，以确保各模态之间的协同整合。（ii）丰富的文本语义矫正模块，该模块采用双向文本特征生成来细化多模态特征对齐，从而增强对语言及其在物体检测中应用的理解。我们在常见的跨域物体检测基准上评估了所提出的方法，并证明了其显著优于现有的少样本物体检测方法。 

---
# Audio Visual Segmentation Through Text Embeddings 

**Title (ZH)**: 通过文本嵌入进行音视频分割 

**Authors**: Kyungbok Lee, You Zhang, Zhiyao Duan  

**Link**: [PDF](https://arxiv.org/pdf/2502.16359)  

**Abstract**: The goal of Audio-Visual Segmentation (AVS) is to localize and segment the sounding source objects from the video frames. Researchers working on AVS suffer from limited datasets because hand-crafted annotation is expensive. Recent works attempt to overcome the challenge of limited data by leveraging the segmentation foundation model, SAM, prompting it with audio to enhance its ability to segment sounding source objects. While this approach alleviates the model's burden on understanding visual modality by utilizing pre-trained knowledge of SAM, it does not address the fundamental challenge of the limited dataset for learning audio-visual relationships. To address these limitations, we propose \textbf{AV2T-SAM}, a novel framework that bridges audio features with the text embedding space of pre-trained text-prompted SAM. Our method leverages multimodal correspondence learned from rich text-image paired datasets to enhance audio-visual alignment. Furthermore, we introduce a novel feature, $\mathbf{\textit{\textbf{f}}_{CLIP} \odot \textit{\textbf{f}}_{CLAP}}$, which emphasizes shared semantics of audio and visual modalities while filtering irrelevant noise. Experiments on the AVSBench dataset demonstrate state-of-the-art performance on both datasets of AVSBench. Our approach outperforms existing methods by effectively utilizing pretrained segmentation models and cross-modal semantic alignment. 

**Abstract (ZH)**: 基于视听分割的音频-视觉分割（AV2T-SAM）框架：结合音频特征与预训练文本指导SAM的文本嵌入空间 

---
# Understanding the Emergence of Multimodal Representation Alignment 

**Title (ZH)**: 多模态表示对齐的 emergence 机制理解 

**Authors**: Megan Tjandrasuwita, Chanakya Ekbote, Liu Ziyin, Paul Pu Liang  

**Link**: [PDF](https://arxiv.org/pdf/2502.16282)  

**Abstract**: Multimodal representation learning is fundamentally about transforming incomparable modalities into comparable representations. While prior research primarily focused on explicitly aligning these representations through targeted learning objectives and model architectures, a recent line of work has found that independently trained unimodal models of increasing scale and performance can become implicitly aligned with each other. These findings raise fundamental questions regarding the emergence of aligned representations in multimodal learning. Specifically: (1) when and why does alignment emerge implicitly? and (2) is alignment a reliable indicator of performance? Through a comprehensive empirical investigation, we demonstrate that both the emergence of alignment and its relationship with task performance depend on several critical data characteristics. These include, but are not necessarily limited to, the degree of similarity between the modalities and the balance between redundant and unique information they provide for the task. Our findings suggest that alignment may not be universally beneficial; rather, its impact on performance varies depending on the dataset and task. These insights can help practitioners determine whether increasing alignment between modalities is advantageous or, in some cases, detrimental to achieving optimal performance. Code is released at this https URL. 

**Abstract (ZH)**: 多模态表示学习本质上是将不可比较的模态转换为可比较的表示。尽管先前的研究主要通过目标学习任务和模型架构来显式对齐这些表示，近期的一项研究成果发现，独立训练的、规模和性能不断增加的单模态模型可以隐式对齐。这些发现引发了关于多模态学习中对齐表示出现的基本问题，具体包括：（1）在何时以及为何种情况下对齐会隐式出现？（2）对齐是否是性能可靠的指标？通过全面的实证研究，我们证明了对齐的出现及其与任务性能的关系取决于多种关键的数据特征，这些特征包括模态之间的相似度以及它们为任务提供的冗余信息与独特信息之间的平衡。我们的研究结果表明，对齐并不一定是普遍有益的，其对性能的影响取决于数据集和任务。这些见解可以帮助从业者判断增加模态之间的对齐是否对实现最佳性能有利，在某些情况下可能是不利的。代码发布在该网址：https://。 

---
