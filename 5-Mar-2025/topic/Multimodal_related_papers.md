# Resource-Efficient Affordance Grounding with Complementary Depth and Semantic Prompts 

**Title (ZH)**: 资源高效的功能 grounding 方法：互补的深度信息和语义提示 

**Authors**: Yizhou Huang, Fan Yang, Guoliang Zhu, Gen Li, Hao Shi, Yukun Zuo, Wenrui Chen, Zhiyong Li, Kailun Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.02600)  

**Abstract**: Affordance refers to the functional properties that an agent perceives and utilizes from its environment, and is key perceptual information required for robots to perform actions. This information is rich and multimodal in nature. Existing multimodal affordance methods face limitations in extracting useful information, mainly due to simple structural designs, basic fusion methods, and large model parameters, making it difficult to meet the performance requirements for practical deployment. To address these issues, this paper proposes the BiT-Align image-depth-text affordance mapping framework. The framework includes a Bypass Prompt Module (BPM) and a Text Feature Guidance (TFG) attention selection mechanism. BPM integrates the auxiliary modality depth image directly as a prompt to the primary modality RGB image, embedding it into the primary modality encoder without introducing additional encoders. This reduces the model's parameter count and effectively improves functional region localization accuracy. The TFG mechanism guides the selection and enhancement of attention heads in the image encoder using textual features, improving the understanding of affordance characteristics. Experimental results demonstrate that the proposed method achieves significant performance improvements on public AGD20K and HICO-IIF datasets. On the AGD20K dataset, compared with the current state-of-the-art method, we achieve a 6.0% improvement in the KLD metric, while reducing model parameters by 88.8%, demonstrating practical application values. The source code will be made publicly available at this https URL. 

**Abstract (ZH)**: affordance映射框架：BiT-Align图像-深度-文本对齐 

---
# Attention Bootstrapping for Multi-Modal Test-Time Adaptation 

**Title (ZH)**: 多模态测试时自适应的注意力强化方法 

**Authors**: Yusheng Zhao, Junyu Luo, Xiao Luo, Jinsheng Huang, Jingyang Yuan, Zhiping Xiao, Ming Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.02221)  

**Abstract**: Test-time adaptation aims to adapt a well-trained model to potential distribution shifts at test time using only unlabeled test data, without access to the original training data. While previous efforts mainly focus on a single modality, test-time distribution shift in the multi-modal setting is more complex and calls for new solutions. This paper tackles the problem of multi-modal test-time adaptation by proposing a novel method named Attention Bootstrapping with Principal Entropy Minimization (ABPEM). We observe that test-time distribution shift causes misalignment across modalities, leading to a large gap between intra-modality discrepancies (measured by self-attention) and inter-modality discrepancies (measured by cross-attention). We name this the attention gap. This attention gap widens with more severe distribution shifts, hindering effective modality fusion. To mitigate this attention gap and encourage better modality fusion, we propose attention bootstrapping that promotes cross-attention with the guidance of self-attention. Moreover, to reduce the gradient noise in the commonly-used entropy minimization, we adopt principal entropy minimization, a refinement of entropy minimization that reduces gradient noise by focusing on the principal parts of entropy, excluding less reliable gradient information. Extensive experiments on the benchmarks validate the effectiveness of the proposed ABPEM in comparison with competing baselines. 

**Abstract (ZH)**: 多模态测试时自适应：基于主熵最小化的注意力-bootstrap方法（Attention Bootstrapping with Principal Entropy Minimization for Multi-modal Test-time Adaptation） 

---
# Nexus-O: An Omni-Perceptive And -Interactive Model for Language, Audio, And Vision 

**Title (ZH)**: Nexus-O：一种综合感知和交互的语言、音频和视觉模型 

**Authors**: Che Liu, Yingji Zhang, Dong Zhang, Weijie Zhang, Chenggong Gong, Haohan Li, Yu Lu, Shilin Zhou, Yue Lu, Ziliang Gan, Ziao Wang, Junwei Liao, Haipang Wu, Ji Liu, André Freitas, Qifan Wang, Zenglin Xu, Rongjuncheng Zhang, Yong Dai  

**Link**: [PDF](https://arxiv.org/pdf/2503.01879)  

**Abstract**: Human beings perceive the real world through a spectrum of sensory modalities, encompassing auditory, visual, and linguistic faculties. The journey towards achieving Artificial General Intelligence (AGI) necessitates the development of models that can emulate these multifaceted perceptual capabilities and comprehensively understand these diversified data. To this end, we introduce \textbf{Nexus-O}, an industry-level \textbf{omni-perceptive and -interactive} model capable of efficiently processing Audio, Image, Video, and Text data in any combination and output audio/text in an end-to-end way. We systematically investigate Nexus-O by addressing three key research questions: First, how can models be efficiently designed and trained to achieve tri-modal alignment, understanding and reasoning capabilities across multiple modalities? Second, what approaches can be implemented to evaluate tri-modal model robustness, ensuring reliable performance and applicability in real-world scenarios? Third, what strategies can be employed to curate and obtain high-quality, real-life scenario speech datasets? For the first question, we design and pre-train Nexus-O based on the vision-language model, rather than the language model. By pre-training the model over high-quality synthetic audio data, our model is capable of tri-modal perception and interaction. For the second question, we introduce a new audio testbed, Nexus-O-audio, comprising diverse Automatic Speech Recognition (ASR) samples, spanning various real-world scenarios, such as corporate meetings and live stream. For the third question, we design the speech data synthesis pipeline to obtain high-quality speech training datasets, covering various real-world scenarios. Comprehensive experimentation and an in-depth analysis of tri-modal alignment over latent space demonstrate the advantages of our model on downstream tasks. 

**Abstract (ZH)**: 人类通过听觉、视觉和语言等多种感知模态来认知真实世界。实现通用人工智能（AGI）的旅程需要开发能够模拟这些多维感知能力并全面理解这些多样化数据的模型。为此，我们介绍了一种工业级的**全感知全交互**模型**Nexus-O**，该模型能够高效地处理音频、图像、视频和文本数据的任意组合，并以端到端的方式输出音频/文本。我们系统地探讨了Nexus-O，针对三个关键研究问题进行了研究：首先，如何设计和训练模型以实现跨多种模态的三模态对齐、理解和推理能力？其次，采用什么方法评估三模态模型的鲁棒性，确保其在实际场景中的可靠表现和适用性？第三，如何收集和获得高质量的真实场景语音数据集？对于第一个问题，我们基于视觉-语言模型而非语言模型设计并预训练了Nexus-O，通过在高质量合成音频数据上进行预训练，使模型具备三模态感知和交互的能力。对于第二个问题，我们引入了一个新的音频测试平台Nexus-O-audio，包含多种自动语音识别（ASR）样本，覆盖各种实际场景，如企业会议和直播。对于第三个问题，我们设计了语音数据合成流水线，以获得高质量的语音训练数据集，涵盖各种实际场景。全面的实验和潜空间三模态对齐的深入分析证明了该模型在下游任务中的优势。 

---
# Multimodal Deep Learning for Subtype Classification in Breast Cancer Using Histopathological Images and Gene Expression Data 

**Title (ZH)**: 基于组织病理图像和基因表达数据的乳腺癌亚型分类的多模态深度学习 

**Authors**: Amin Honarmandi Shandiz  

**Link**: [PDF](https://arxiv.org/pdf/2503.02849)  

**Abstract**: Molecular subtyping of breast cancer is crucial for personalized treatment and prognosis. Traditional classification approaches rely on either histopathological images or gene expression profiling, limiting their predictive power. In this study, we propose a deep multimodal learning framework that integrates histopathological images and gene expression data to classify breast cancer into this http URL and this http URL / Her2 subtypes. Our approach employs a ResNet-50 model for image feature extraction and fully connected layers for gene expression processing, with a cross-attention fusion mechanism to enhance modality interaction. We conduct extensive experiments using five-fold cross-validation, demonstrating that our multimodal integration outperforms unimodal approaches in terms of classification accuracy, precision-recall AUC, and F1-score. Our findings highlight the potential of deep learning for robust and interpretable breast cancer subtype classification, paving the way for improved clinical decision-making. 

**Abstract (ZH)**: 基于多模态学习的乳腺癌分子亚型分类对于个性化治疗和预后至关重要。传统分类方法依赖于组织病理学图像或基因表达谱，限制了其预测能力。在本研究中，我们提出了一种深度多模态学习框架，将组织病理学图像和基因表达数据结合起来，将乳腺癌分类为luminal A/luminal B /Her2亚型。我们的方法使用ResNet-50模型进行图像特征提取，并使用全连接层处理基因表达数据，采用交叉注意力融合机制以增强模态间交互。通过五折交叉验证进行广泛的实验，结果显示，我们的多模态集成在分类准确性、精确召回AUC和F1分数等方面优于单一模态方法。我们的研究结果强调了深度学习在稳健且可解释的乳腺癌亚型分类中的潜力，为改进临床决策铺平了道路。 

---
# Developing a PET/CT Foundation Model for Cross-Modal Anatomical and Functional Imaging 

**Title (ZH)**: 开发一种PET/CT跨模态解剖与功能成像基础模型 

**Authors**: Yujin Oh, Robert Seifert, Yihan Cao, Christoph Clement, Justin Ferdinandus, Constantin Lapa, Alessandro Liebich, Michelle Amon, Johanna Enke, Sifan Song, Runqi Meng, Fang Zeng, Ning Guo, Xiang Li, Pedram Heidari, Axel Rominger, Kuangyu Shi, Quanzheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.02824)  

**Abstract**: In oncology, Positron Emission Tomography-Computed Tomography (PET/CT) is widely used in cancer diagnosis, staging, and treatment monitoring, as it combines anatomical details from CT with functional metabolic activity and molecular marker expression information from PET. However, existing artificial intelligence-driven PET/CT analyses rely predominantly on task-specific models trained from scratch or on limited datasets, limiting their generalizability and robustness. To address this, we propose a foundation model approach specifically designed for multimodal PET/CT imaging. We introduce the Cross-Fraternal Twin Masked Autoencoder (FratMAE), a novel framework that effectively integrates whole-body anatomical and functional or molecular information. FratMAE employs separate Vision Transformer (ViT) encoders for PET and CT scans, along with cross-attention decoders that enable synergistic interactions between modalities during masked autoencoder training. Additionally, it incorporates textual metadata to enhance PET representation learning. By pre-training on PET/CT datasets, FratMAE captures intricate cross-modal relationships and global uptake patterns, achieving superior performance on downstream tasks and demonstrating its potential as a generalizable foundation model. 

**Abstract (ZH)**: 在肿瘤学中，正电子发射断层扫描-计算机断层扫描（PET/CT）广泛用于癌症诊断、分期和治疗监测，因为它结合了CT的解剖细节和PET的功能代谢活动及分子标志物表达信息。然而，现有的基于人工智能的PET/CT分析主要依赖于从头训练的任务特定模型或有限的数据集，限制了其泛化能力和 robustness。为此，我们提出了一种专门为多模态PET/CT成像设计的基础模型方法。我们引入了跨同胞双胞胎掩蔽自动编码器（FratMAE）这一新颖框架，能够有效地整合全身解剖和功能或分子信息。FratMAE使用分别针对PET和CT扫描的视觉变换器（ViT）编码器，以及跨注意力解码器，在掩蔽自动编码器训练过程中实现模态间的协同交互。此外，它还整合了文本元数据以增强PET表征学习。通过在PET/CT数据集上进行预训练，FratMAE捕获了复杂的跨模态关系和全球摄取模式，实现了下游任务的优越性能，并展示了其作为泛化基础模型的潜力。 

---
# A Multimodal Symphony: Integrating Taste and Sound through Generative AI 

**Title (ZH)**: 多模态交响曲：通过生成式AI整合味觉与听觉 

**Authors**: Matteo Spanio, Massimiliano Zampini, Antonio Rodà, Franco Pierucci  

**Link**: [PDF](https://arxiv.org/pdf/2503.02823)  

**Abstract**: In recent decades, neuroscientific and psychological research has traced direct relationships between taste and auditory perceptions. This article explores multimodal generative models capable of converting taste information into music, building on this foundational research. We provide a brief review of the state of the art in this field, highlighting key findings and methodologies. We present an experiment in which a fine-tuned version of a generative music model (MusicGEN) is used to generate music based on detailed taste descriptions provided for each musical piece. The results are promising: according the participants' ($n=111$) evaluation, the fine-tuned model produces music that more coherently reflects the input taste descriptions compared to the non-fine-tuned model. This study represents a significant step towards understanding and developing embodied interactions between AI, sound, and taste, opening new possibilities in the field of generative AI. We release our dataset, code and pre-trained model at: this https URL. 

**Abstract (ZH)**: 近几十年来，神经科学和心理学研究已揭示了味觉与听觉感知之间的直接关系。本文探讨了能够将味觉信息转换为音乐的多模态生成模型，建立在这一基础研究之上。我们简要回顾了该领域的最新进展，突出显示了关键发现和研究方法。我们提出了一项实验，在该实验中，对生成音乐模型（MusicGEN）进行了微调，并根据为每首音乐作品提供的详细味觉描述生成音乐。结果显示：根据111名参与者的意见评价，微调后的模型生成的音乐更连贯地反映了输入的味觉描述，与未经微调的模型相比更为一致。本文代表了理解并发展人工智能、声音和味觉之间实体交互的重要一步，为生成人工智能领域开辟了新的可能性。我们发布了我们的数据集、代码和预训练模型：[这个链接](this https URL)。 

---
# Seeing is Understanding: Unlocking Causal Attention into Modality-Mutual Attention for Multimodal LLMs 

**Title (ZH)**: 所见即所解：解锁模态互注意中的因果注意力以改善多模态LLMs 

**Authors**: Wei-Yao Wang, Zhao Wang, Helen Suzuki, Yoshiyuki Kobayashi  

**Link**: [PDF](https://arxiv.org/pdf/2503.02597)  

**Abstract**: Recent Multimodal Large Language Models (MLLMs) have demonstrated significant progress in perceiving and reasoning over multimodal inquiries, ushering in a new research era for foundation models. However, vision-language misalignment in MLLMs has emerged as a critical challenge, where the textual responses generated by these models are not factually aligned with the given text-image inputs. Existing efforts to address vision-language misalignment have focused on developing specialized vision-language connectors or leveraging visual instruction tuning from diverse domains. In this paper, we tackle this issue from a fundamental yet unexplored perspective by revisiting the core architecture of MLLMs. Most MLLMs are typically built on decoder-only LLMs consisting of a causal attention mechanism, which limits the ability of earlier modalities (e.g., images) to incorporate information from later modalities (e.g., text). To address this problem, we propose AKI, a novel MLLM that unlocks causal attention into modality-mutual attention (MMA) to enable image tokens to attend to text tokens. This simple yet effective design allows AKI to achieve superior performance in 12 multimodal understanding benchmarks (+7.2% on average) without introducing additional parameters and increasing training time. Our MMA design is intended to be generic, allowing for application across various modalities, and scalable to accommodate diverse multimodal scenarios. The code is publicly available at this https URL, and we will release our AKI-4B model to encourage further advancements in MLLMs across various directions. 

**Abstract (ZH)**: 近期的多模态大型语言模型在处理和推理多模态查询方面取得了显著进步，开启了基础模型研究的新时代。然而，多模态大型语言模型中的视觉-语言不一致性已成为一个关键挑战，模型生成的文本响应与给定的文本-图像输入不一致。现有解决视觉-语言不一致性的方法主要集中在开发专门的视觉-语言连接器或利用来自不同领域的视觉指令调优上。本文从一个基础但未被充分探索的角度出发，重新审视多模态大型语言模型的核心架构。大多数多模态大型语言模型通常基于仅解码器的大规模语言模型，其中包含因果注意力机制，这限制了早期模态（如图像）从后续模态（如文本）获取信息的能力。为了解决这一问题，我们提出了一种名为AKI的新颖多模态大型语言模型，解锁因果注意力到模态互注意力（MMA），使图像令牌能够关注文本令牌。这一简单而有效的设计使AKI在12个多模态理解基准测试中表现卓越（平均提高7.2%），且无需增加参数数量和提高训练时间。我们的MMA设计旨在通用且可扩展，适用于各种模态，并可适应不同的多模态场景。有关代码请访问此网址，我们将发布AKI-4B模型，以促进在各种方向上进一步推进多模态大型语言模型的研究。 

---
# Semi-Supervised Audio-Visual Video Action Recognition with Audio Source Localization Guided Mixup 

**Title (ZH)**: 基于音频源定位引导Mixup的半监督音频-视觉视频动作识别 

**Authors**: Seokun Kang, Taehwan Kim  

**Link**: [PDF](https://arxiv.org/pdf/2503.02284)  

**Abstract**: Video action recognition is a challenging but important task for understanding and discovering what the video does. However, acquiring annotations for a video is costly, and semi-supervised learning (SSL) has been studied to improve performance even with a small number of labeled data in the task. Prior studies for semi-supervised video action recognition have mostly focused on using single modality - visuals - but the video is multi-modal, so utilizing both visuals and audio would be desirable and improve performance further, which has not been explored well. Therefore, we propose audio-visual SSL for video action recognition, which uses both visual and audio together, even with quite a few labeled data, which is challenging. In addition, to maximize the information of audio and video, we propose a novel audio source localization-guided mixup method that considers inter-modal relations between video and audio modalities. In experiments on UCF-51, Kinetics-400, and VGGSound datasets, our model shows the superior performance of the proposed semi-supervised audio-visual action recognition framework and audio source localization-guided mixup. 

**Abstract (ZH)**: 基于音频-视觉半监督学习的视频动作识别 

---
# Words or Vision: Do Vision-Language Models Have Blind Faith in Text? 

**Title (ZH)**: 词语or视觉：视觉语言模型对文本是否盲目信赖？ 

**Authors**: Ailin Deng, Tri Cao, Zhirui Chen, Bryan Hooi  

**Link**: [PDF](https://arxiv.org/pdf/2503.02199)  

**Abstract**: Vision-Language Models (VLMs) excel in integrating visual and textual information for vision-centric tasks, but their handling of inconsistencies between modalities is underexplored. We investigate VLMs' modality preferences when faced with visual data and varied textual inputs in vision-centered settings. By introducing textual variations to four vision-centric tasks and evaluating ten Vision-Language Models (VLMs), we discover a \emph{``blind faith in text''} phenomenon: VLMs disproportionately trust textual data over visual data when inconsistencies arise, leading to significant performance drops under corrupted text and raising safety concerns. We analyze factors influencing this text bias, including instruction prompts, language model size, text relevance, token order, and the interplay between visual and textual certainty. While certain factors, such as scaling up the language model size, slightly mitigate text bias, others like token order can exacerbate it due to positional biases inherited from language models. To address this issue, we explore supervised fine-tuning with text augmentation and demonstrate its effectiveness in reducing text bias. Additionally, we provide a theoretical analysis suggesting that the blind faith in text phenomenon may stem from an imbalance of pure text and multi-modal data during training. Our findings highlight the need for balanced training and careful consideration of modality interactions in VLMs to enhance their robustness and reliability in handling multi-modal data inconsistencies. 

**Abstract (ZH)**: 视觉-语言模型（VLMs）在视觉中心任务中擅长整合视觉和文本信息，但它们处理不同模态之间不一致性的方法尚未充分探讨。我们调查了在面对视觉数据和不同文本输入时，视觉中心设置中VLMs对不同模态的偏好。通过向四个视觉中心任务引入文本变化并评估十种视觉-语言模型（VLMs），我们发现了一种“过度依赖文本”的现象：当出现不一致性时，VLMs过度信任文本数据而忽视视觉数据，导致在文本受污染时性能显著下降，并引发安全关切。我们分析了影响这一文本偏差的因素，包括指令提示、语言模型规模、文本相关性、标记顺序以及视觉和文本一致性的互动。虽然某些因素，如扩大语言模型规模，可以略微减轻文本偏差，但其他因素，如标记顺序，却可能因继承自语言模型的位置偏差而加剧偏差。为了解决这一问题，我们探索了带有文本增强的数据监督微调方法，并展示了其在减少文本偏差方面的有效性。此外，我们提供了一种理论分析，表明这一“过度依赖文本”现象可能源于训练过程中纯文本和多模态数据之间失衡。我们的研究结果强调了平衡训练和仔细考虑模态交互的重要性，以增强VLMs在处理多模态数据不一致性时的稳健性和可靠性。 

---
# DivPrune: Diversity-based Visual Token Pruning for Large Multimodal Models 

**Title (ZH)**: DivPrune：基于多样性的视觉词元剪枝用于大型多模态模型 

**Authors**: Saeed Ranjbar Alvar, Gursimran Singh, Mohammad Akbari, Yong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.02175)  

**Abstract**: Large Multimodal Models (LMMs) have emerged as powerful models capable of understanding various data modalities, including text, images, and videos. LMMs encode both text and visual data into tokens that are then combined and processed by an integrated Large Language Model (LLM). Including visual tokens substantially increases the total token count, often by thousands. The increased input length for LLM significantly raises the complexity of inference, resulting in high latency in LMMs. To address this issue, token pruning methods, which remove part of the visual tokens, are proposed. The existing token pruning methods either require extensive calibration and fine-tuning or rely on suboptimal importance metrics which results in increased redundancy among the retained tokens. In this paper, we first formulate token pruning as Max-Min Diversity Problem (MMDP) where the goal is to select a subset such that the diversity among the selected {tokens} is maximized. Then, we solve the MMDP to obtain the selected subset and prune the rest. The proposed method, DivPrune, reduces redundancy and achieves the highest diversity of the selected tokens. By ensuring high diversity, the selected tokens better represent the original tokens, enabling effective performance even at high pruning ratios without requiring fine-tuning. Extensive experiments with various LMMs show that DivPrune achieves state-of-the-art accuracy over 16 image- and video-language datasets. Additionally, DivPrune reduces both the end-to-end latency and GPU memory usage for the tested models. The code is available $\href{this https URL}{\text{here}}$. 

**Abstract (ZH)**: 大型多模态模型中的token剪枝方法：最大化最小多样性的剪枝方法（DivPrune） 

---
# Abn-BLIP: Abnormality-aligned Bootstrapping Language-Image Pre-training for Pulmonary Embolism Diagnosis and Report Generation from CTPA 

**Title (ZH)**: Abn-BLIP: 与异常对齐的引导语言-图像预训练在CTPA肺栓塞诊断与报告生成中的应用 

**Authors**: Zhusi Zhong, Yuli Wang, Lulu Bi, Zhuoqi Ma, Sun Ho Ahn, Christopher J. Mullin, Colin F. Greineder, Michael K. Atalay, Scott Collins, Grayson L. Baird, Cheng Ting Lin, Webster Stayman, Todd M. Kolb, Ihab Kamel, Harrison X. Bai, Zhicheng Jiao  

**Link**: [PDF](https://arxiv.org/pdf/2503.02034)  

**Abstract**: Medical imaging plays a pivotal role in modern healthcare, with computed tomography pulmonary angiography (CTPA) being a critical tool for diagnosing pulmonary embolism and other thoracic conditions. However, the complexity of interpreting CTPA scans and generating accurate radiology reports remains a significant challenge. This paper introduces Abn-BLIP (Abnormality-aligned Bootstrapping Language-Image Pretraining), an advanced diagnosis model designed to align abnormal findings to generate the accuracy and comprehensiveness of radiology reports. By leveraging learnable queries and cross-modal attention mechanisms, our model demonstrates superior performance in detecting abnormalities, reducing missed findings, and generating structured reports compared to existing methods. Our experiments show that Abn-BLIP outperforms state-of-the-art medical vision-language models and 3D report generation methods in both accuracy and clinical relevance. These results highlight the potential of integrating multimodal learning strategies for improving radiology reporting. The source code is available at this https URL. 

**Abstract (ZH)**: 医学影像在现代医疗保健中扮演着关键角色，计算机断层扫描肺动脉造影（CTPA）是诊断肺栓塞和其他胸腔疾病的重要工具。然而，CTPA扫描的解释复杂性和生成准确的放射学报告仍是一项重大挑战。本文介绍了一种先进的诊断模型Abn-BLIP（异常对齐的自举语言-图像预训练），该模型旨在将异常发现对齐以生成放射学报告的准确性和全面性。通过利用可学习的查询和跨模态注意力机制，我们的模型在检测异常、减少遗漏发现以及生成结构化报告方面表现出色，优于现有方法。实验结果表明，Abn-BLIP在准确性和临床相关性方面均优于最先进的医疗视觉-语言模型和3D报告生成方法。这些结果突显了结合多模态学习策略以提高放射学报告的质量的潜力。源代码可在以下链接获取。 

---
# Recurrence-Enhanced Vision-and-Language Transformers for Robust Multimodal Document Retrieval 

**Title (ZH)**: 增强循环机制的视觉-语言变换器for稳健的多模态文档检索 

**Authors**: Davide Caffagni, Sara Sarto, Marcella Cornia, Lorenzo Baraldi, Rita Cucchiara  

**Link**: [PDF](https://arxiv.org/pdf/2503.01980)  

**Abstract**: Cross-modal retrieval is gaining increasing efficacy and interest from the research community, thanks to large-scale training, novel architectural and learning designs, and its application in LLMs and multimodal LLMs. In this paper, we move a step forward and design an approach that allows for multimodal queries, composed of both an image and a text, and can search within collections of multimodal documents, where images and text are interleaved. Our model, ReT, employs multi-level representations extracted from different layers of both visual and textual backbones, both at the query and document side. To allow for multi-level and cross-modal understanding and feature extraction, ReT employs a novel Transformer-based recurrent cell that integrates both textual and visual features at different layers, and leverages sigmoidal gates inspired by the classical design of LSTMs. Extensive experiments on M2KR and M-BEIR benchmarks show that ReT achieves state-of-the-art performance across diverse settings. Our source code and trained models are publicly available at this https URL. 

**Abstract (ZH)**: 跨模态检索随着大规模训练、新颖的架构和学习设计的进步，以及在大型语言模型和多模态大型语言模型中的应用，正日益显示出其有效性并引起了研究社区的兴趣。本文在此基础上提出了一种方法，该方法能够处理由图像和文本组成的多模态查询，并能在图像和文本交织的多模态文档集合中进行搜索。我们的模型ReT利用来自视觉和文本骨干网络不同层的多级表示，应用于查询和文档两侧。为实现多层次和跨模态的理解和特征提取，ReT采用了一种新颖的基于Transformer的递归单元，该单元在不同层中整合了文本和视觉特征，并借鉴了经典LSTM设计中的sigmoid门控机制。在M2KR和M-BEIR基准上的广泛实验表明，ReT在各种设置中均实现了最先进的性能。我们的源代码和训练模型已公开在此URL：this https URL。 

---
# What are You Looking at? Modality Contribution in Multimodal Medical Deep Learning Methods 

**Title (ZH)**: 你正在关注什么？多模态医疗深度学习方法中的模态贡献 

**Authors**: Christian Gapp, Elias Tappeiner, Martin Welk, Karl Fritscher, Elke Ruth Gizewski, Rainer Schubert  

**Link**: [PDF](https://arxiv.org/pdf/2503.01904)  

**Abstract**: Purpose High dimensional, multimodal data can nowadays be analyzed by huge deep neural networks with little effort. Several fusion methods for bringing together different modalities have been developed. Particularly, in the field of medicine with its presence of high dimensional multimodal patient data, multimodal models characterize the next step. However, what is yet very underexplored is how these models process the source information in detail. Methods To this end, we implemented an occlusion-based both model and performance agnostic modality contribution method that quantitatively measures the importance of each modality in the dataset for the model to fulfill its task. We applied our method to three different multimodal medical problems for experimental purposes. Results Herein we found that some networks have modality preferences that tend to unimodal collapses, while some datasets are imbalanced from the ground up. Moreover, we could determine a link between our metric and the performance of single modality trained nets. Conclusion The information gain through our metric holds remarkable potential to improve the development of multimodal models and the creation of datasets in the future. With our method we make a crucial contribution to the field of interpretability in deep learning based multimodal research and thereby notably push the integrability of multimodal AI into clinical practice. Our code is publicly available at this https URL. 

**Abstract (ZH)**: 目的：现有的大型深度神经网络可以轻松分析高维多模态数据。已经开发出了多种融合不同模态的方法。特别是，在医学领域，由于存在高维多模态患者数据，多模态模型是进一步的发展方向。然而，这些模型如何处理源信息的具体细节仍然很少被探索。方法：为此，我们实现了一种基于遮罩的、与模型和性能无关的模态贡献方法，用于定量测量数据集中每个模态对于模型完成任务的重要性。我们针对三种不同的多模态医学问题进行了实验研究。结果：我们发现，某些网络具有模态偏好，倾向于单一模态的收敛；某些数据集从一开始就存在不平衡性。此外，我们能够确定我们的指标与单模态训练网络性能之间的联系。结论：通过我们的指标获得的信息增益对未来多模态模型的发展和数据集的创建具有显著潜力。我们的方法在基于深度学习的多模态研究的可解释性领域做出了关键性贡献，从而显著推动了多模态AI在临床实践中的集成。我们的代码已在以下网址公开 доступ于此：this https URL。 

---
