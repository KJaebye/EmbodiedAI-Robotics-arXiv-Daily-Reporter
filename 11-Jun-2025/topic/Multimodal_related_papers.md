# Efficient Medical Vision-Language Alignment Through Adapting Masked Vision Models 

**Title (ZH)**: 通过适应掩蔽视觉模型实现高效的医学视图-语言对齐 

**Authors**: Chenyu Lian, Hong-Yu Zhou, Dongyun Liang, Jing Qin, Liansheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.08990)  

**Abstract**: Medical vision-language alignment through cross-modal contrastive learning shows promising performance in image-text matching tasks, such as retrieval and zero-shot classification. However, conventional cross-modal contrastive learning (CLIP-based) methods suffer from suboptimal visual representation capabilities, which also limits their effectiveness in vision-language alignment. In contrast, although the models pretrained via multimodal masked modeling struggle with direct cross-modal matching, they excel in visual representation. To address this contradiction, we propose ALTA (ALign Through Adapting), an efficient medical vision-language alignment method that utilizes only about 8% of the trainable parameters and less than 1/5 of the computational consumption required for masked record modeling. ALTA achieves superior performance in vision-language matching tasks like retrieval and zero-shot classification by adapting the pretrained vision model from masked record modeling. Additionally, we integrate temporal-multiview radiograph inputs to enhance the information consistency between radiographs and their corresponding descriptions in reports, further improving the vision-language alignment. Experimental evaluations show that ALTA outperforms the best-performing counterpart by over 4% absolute points in text-to-image accuracy and approximately 6% absolute points in image-to-text retrieval accuracy. The adaptation of vision-language models during efficient alignment also promotes better vision and language understanding. Code is publicly available at this https URL. 

**Abstract (ZH)**: 医学视觉-语言对齐通过跨模态对比学习在图像-文本匹配任务中表现出色，但传统跨模态对比学习（如CLIP）方法在视觉表示能力上有不足，限制了其在视觉-语言对齐中的效果。相比之下，虽然通过多模态掩码建模预训练的模型在直接跨模态匹配上存在问题，但在视觉表示上表现出色。为了解决这一矛盾，我们提出了一种高效的医学视觉-语言对齐方法ALTA（通过适应对齐），该方法仅使用约8%的可训练参数和少于掩码记录建模所需计算量的1/5。ALTA通过适应从掩码记录建模预训练的视觉模型，在视觉-语言匹配任务如检索和零样本分类中取得了优异性能。此外，我们整合了时间多视图X线输入，以增强医学报告中的X线图像与其描述之间的一致性，进一步提高视觉-语言对齐。实验评估显示，ALTA在文本到图像准确性和图像到文本检索准确率上分别超过当前最优方法4%和约6%的绝对值。通过有效对齐促进视觉和语言理解。源代码已公开。 

---
# Multimodal Representation Alignment for Cross-modal Information Retrieval 

**Title (ZH)**: 多模态表示对齐在跨模态信息检索中的应用 

**Authors**: Fan Xu, Luis A. Leiva  

**Link**: [PDF](https://arxiv.org/pdf/2506.08774)  

**Abstract**: Different machine learning models can represent the same underlying concept in different ways. This variability is particularly valuable for in-the-wild multimodal retrieval, where the objective is to identify the corresponding representation in one modality given another modality as input. This challenge can be effectively framed as a feature alignment problem. For example, given a sentence encoded by a language model, retrieve the most semantically aligned image based on features produced by an image encoder, or vice versa. In this work, we first investigate the geometric relationships between visual and textual embeddings derived from both vision-language models and combined unimodal models. We then align these representations using four standard similarity metrics as well as two learned ones, implemented via neural networks. Our findings indicate that the Wasserstein distance can serve as an informative measure of the modality gap, while cosine similarity consistently outperforms alternative metrics in feature alignment tasks. Furthermore, we observe that conventional architectures such as multilayer perceptrons are insufficient for capturing the complex interactions between image and text representations. Our study offers novel insights and practical considerations for researchers working in multimodal information retrieval, particularly in real-world, cross-modal applications. 

**Abstract (ZH)**: 不同的机器学习模型可以以不同的方式表示相同的潜在概念。这种多样性特别适用于实时多模态检索，在这种检索中，目标是根据输入的另一种模态识别相应的表示。这一挑战可以有效地被框架为特征对齐问题。例如，给定由语言模型编码的一句话，基于图像编码器产生的特征检索最具有语义对齐的图像，反之亦然。在本文中，我们首先研究来自视觉-语言模型和联合单模态模型的视觉和文本嵌入之间的几何关系，然后使用四种标准相似度度量以及两种通过神经网络实现的学习度量来对这些表示进行对齐。我们的研究结果表明，Wasserstein距离可以作为模态差距的一个有用度量，而余弦相似度在特征对齐任务中始终优于其他度量。此外，我们观察到，传统的架构如多层感知机不足以捕捉图像和文本表示之间的复杂交互。我们的研究为从事多模态信息检索的研究人员，特别是在现实世界、跨模态应用方面，提供了新的见解和实际考虑。 

---
# CoMuMDR: Code-mixed Multi-modal Multi-domain corpus for Discourse paRsing in conversations 

**Title (ZH)**: CoMuMDR：代码混合多模态多领域语料库及其在对话话语解析中的应用 

**Authors**: Divyaksh Shukla, Ritesh Baviskar, Dwijesh Gohil, Aniket Tiwari, Atul Shree, Ashutosh Modi  

**Link**: [PDF](https://arxiv.org/pdf/2506.08504)  

**Abstract**: Discourse parsing is an important task useful for NLU applications such as summarization, machine comprehension, and emotion recognition. The current discourse parsing datasets based on conversations consists of written English dialogues restricted to a single domain. In this resource paper, we introduce CoMuMDR: Code-mixed Multi-modal Multi-domain corpus for Discourse paRsing in conversations. The corpus (code-mixed in Hindi and English) has both audio and transcribed text and is annotated with nine discourse relations. We experiment with various SoTA baseline models; the poor performance of SoTA models highlights the challenges of multi-domain code-mixed corpus, pointing towards the need for developing better models for such realistic settings. 

**Abstract (ZH)**: Code-mixed Multi-modal Multi-domain corpus for Discourse Parsing in Conversations 

---
# Re-Thinking the Automatic Evaluation of Image-Text Alignment in Text-to-Image Models 

**Title (ZH)**: 重思文本到图像模型中图像-文本对齐的自动评价 

**Authors**: Huixuan Zhang, Xiaojun Wan  

**Link**: [PDF](https://arxiv.org/pdf/2506.08480)  

**Abstract**: Text-to-image models often struggle to generate images that precisely match textual prompts. Prior research has extensively studied the evaluation of image-text alignment in text-to-image generation. However, existing evaluations primarily focus on agreement with human assessments, neglecting other critical properties of a trustworthy evaluation framework. In this work, we first identify two key aspects that a reliable evaluation should address. We then empirically demonstrate that current mainstream evaluation frameworks fail to fully satisfy these properties across a diverse range of metrics and models. Finally, we propose recommendations for improving image-text alignment evaluation. 

**Abstract (ZH)**: Text-to-image模型常常难以生成与文本提示精确匹配的图像。先前的研究已经广泛研究了文本到图像生成中的图像-文本对齐评估。然而，现有的评估主要侧重于与人类评估的一致性，忽视了可信评估框架的其他关键属性。在本文中，我们首先识别出可靠评估应该解决的两个关键方面。然后，我们通过实证方法证明当前主流的评估框架无法在多种度量标准和模型中全面满足这些属性。最后，我们提出改进图像-文本对齐评估的建议。 

---
# Seeing Voices: Generating A-Roll Video from Audio with Mirage 

**Title (ZH)**: 听见 vozces：从音频生成A-roll视频的Mirage方法 

**Authors**: Aditi Sundararaman, Amogh Adishesha, Andrew Jaegle, Dan Bigioi, Hyoung-Kyu Song, Jon Kyl, Justin Mao, Kevin Lan, Mojtaba Komeili, ShahRukh Athar, Sheila Babayan, Stanislau Beliasau, William Buchwalter  

**Link**: [PDF](https://arxiv.org/pdf/2506.08279)  

**Abstract**: From professional filmmaking to user-generated content, creators and consumers have long recognized that the power of video depends on the harmonious integration of what we hear (the video's audio track) with what we see (the video's image sequence). Current approaches to video generation either ignore sound to focus on general-purpose but silent image sequence generation or address both visual and audio elements but focus on restricted application domains such as re-dubbing. We introduce Mirage, an audio-to-video foundation model that excels at generating realistic, expressive output imagery from scratch given an audio input. When integrated with existing methods for speech synthesis (text-to-speech, or TTS), Mirage results in compelling multimodal video. When trained on audio-video footage of people talking (A-roll) and conditioned on audio containing speech, Mirage generates video of people delivering a believable interpretation of the performance implicit in input audio. Our central technical contribution is a unified method for training self-attention-based audio-to-video generation models, either from scratch or given existing weights. This methodology allows Mirage to retain generality as an approach to audio-to-video generation while producing outputs of superior subjective quality to methods that incorporate audio-specific architectures or loss components specific to people, speech, or details of how images or audio are captured. We encourage readers to watch and listen to the results of Mirage for themselves (see paper and comments for links). 

**Abstract (ZH)**: 从专业 filmmaking 到用户生成内容，创作人和消费者长期认识到，视频的力量在于所听到的声音（音轨）与所看到的画面（图像序列）的和谐整合。当前的视频生成方法要么忽略声音而专注于无声音的图像序列生成，要么同时处理视觉和音频元素但局限于如重新配音等有限的应用领域。我们介绍了 Mirage，一个从音频生成逼真、具表现力的输出图像的基座模型。当与现有的语音合成方法（文本转语音，或 TTS）结合时，Mirage 会产生引人入胜的多模态视频。当使用人们谈话的音视频素材（A- roll）进行训练，并以包含演讲的音频为条件时，Mirage 生成了人们表现输入音频中隐含表演的真实再现的视频。我们的主要技术贡献是一种统一的方法，用于训练基于自注意力的音频到视频生成模型，无论是从头训练还是在现有权重的基础上训练。这种方法使 Mirage 能够保留其作为音频到视频生成方法的通用性，同时生成感官质量优于包含特定于音频的架构或特定于人、语音或图像和音频捕获细节的损失组件的方法的输出。我们鼓励读者亲自观看和听取 Mirage 的结果（详见论文和评论中的链接）。 

---
# Instruction-Tuned Video-Audio Models Elucidate Functional Specialization in the Brain 

**Title (ZH)**: 指令调优的视频-音频模型阐明大脑的功能专门化 

**Authors**: Subba Reddy Oota, Khushbu Pahwa, Prachi Jindal, Satya Sai Srinath Namburi, Maneesh Singh, Tanmoy Chakraborty, Bapi S. Raju, Manish Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2506.08277)  

**Abstract**: Recent voxel-wise multimodal brain encoding studies have shown that multimodal large language models (MLLMs) exhibit a higher degree of brain alignment compared to unimodal models in both unimodal and multimodal stimulus settings. More recently, instruction-tuned multimodal models have shown to generate task-specific representations that align strongly with brain activity. However, prior work evaluating the brain alignment of MLLMs has primarily focused on unimodal settings or relied on non-instruction-tuned multimodal models for multimodal stimuli. To address this gap, we investigated brain alignment, that is, measuring the degree of predictivity of neural activity recorded while participants were watching naturalistic movies (video along with audio) with representations derived from MLLMs. We utilized instruction-specific embeddings from six video and two audio instruction-tuned MLLMs. Experiments with 13 video task-specific instructions show that instruction-tuned video MLLMs significantly outperform non-instruction-tuned multimodal (by 15%) and unimodal models (by 20%). Our evaluation of MLLMs for both video and audio tasks using language-guided instructions shows clear disentanglement in task-specific representations from MLLMs, leading to precise differentiation of multimodal functional processing in the brain. We also find that MLLM layers align hierarchically with the brain, with early sensory areas showing strong alignment with early layers, while higher-level visual and language regions align more with middle to late layers. These findings provide clear evidence for the role of task-specific instructions in improving the alignment between brain activity and MLLMs, and open new avenues for mapping joint information processing in both the systems. We make the code publicly available [this https URL]. 

**Abstract (ZH)**: Recent 多模态脑编码研究表明，多模态大型语言模型（MLLMs）在单模态和多模态刺激设置中都比单模态模型更接近大脑活动。最近，指令调优的多模态模型展示出能生成与脑活动强烈对应的任伺特异性表示。然而，之前评估MLLMs脑部对齐的工作主要集中在单模态设置上，或者依赖于未指令调优的多模态模型来处理多模态刺激。为解决这一不足，我们探讨了脑部对齐，即通过使用来自六个视频和两个音频指令调优的MLLMs表示，在参与者观看自然电影（视频配以音频）时记录的大脑活动预测度进行测量。实验使用13种视频任伺特异性指令表明，指令调优的视频MLLMs显著优于未指令调优的多模态（高15%）和单模态模型（高20%）。我们使用语言指导的指令评估MLLMs在视频和音频任务中的表现，显示出MLLMs在任务特异性表示上的清晰分离，促进了对接多模态脑功能处理的精确区分。我们还发现，MLLMs层按层次结构与大脑对齐，早期感觉区域与早期层强对齐，而高级视觉和语言区域则更多与中间到晚期层对齐。这些发现提供了任务特异性指令在提高脑活动与MLLMs对齐方面的明确证据，并为映射系统中联合信息处理开辟了新途径。我们公开发布了相关代码 [this https URL]。 

---
