# Toward Aligning Human and Robot Actions via Multi-Modal Demonstration Learning 

**Title (ZH)**: 基于多模态示范学习的人机动作对齐 

**Authors**: Azizul Zahid, Jie Fan, Farong Wang, Ashton Dy, Sai Swaminathan, Fei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.11493)  

**Abstract**: Understanding action correspondence between humans and robots is essential for evaluating alignment in decision-making, particularly in human-robot collaboration and imitation learning within unstructured environments. We propose a multimodal demonstration learning framework that explicitly models human demonstrations from RGB video with robot demonstrations in voxelized RGB-D space. Focusing on the "pick and place" task from the RH20T dataset, we utilize data from 5 users across 10 diverse scenes. Our approach combines ResNet-based visual encoding for human intention modeling and a Perceiver Transformer for voxel-based robot action prediction. After 2000 training epochs, the human model reaches 71.67% accuracy, and the robot model achieves 71.8% accuracy, demonstrating the framework's potential for aligning complex, multimodal human and robot behaviors in manipulation tasks. 

**Abstract (ZH)**: 理解人类与机器人之间的动作对应对于评估决策一致性至关重要，特别是在不规则环境中的人类-机器人协作和模仿学习中。我们提出了一种多模态示范学习框架，该框架明确地将人类演示从RGB视频建模到体素化RGB-D空间中的机器人演示。以RH20T数据集中“拿起放置”任务为例，我们利用5名用户在10个不同场景下的数据。我们的方法结合了基于ResNet的视觉编码进行人类意图建模，以及基于体素的机器人动作预测的感知器变换器。经过2000个训练周期后，人类模型的准确率为71.67%，机器人模型的准确率为71.8%，展示了该框架在复杂多模态人类和机器人操作行为对齐中的潜力。 

---
# From Conceptual Data Models to Multimodal Representation 

**Title (ZH)**: 从概念数据模型到多模态表示 

**Authors**: Peter Stockinger  

**Link**: [PDF](https://arxiv.org/pdf/2504.11459)  

**Abstract**: 1) Introduction and Conceptual Framework: This document explores the concept of information design by dividing it into two major practices: defining the meaning of a corpus of textual data and its visual or multimodal representation. It draws on expertise in enriching textual corpora, particularly audiovisual ones, and transforming them into multiple narrative formats. The text highlights a crucial distinction between the semantic content of a domain and the modalities of its graphic expression, illustrating this approach with concepts rooted in structural semiotics and linguistics traditions.
2) Modeling and Conceptual Design:  The article emphasizes the importance of semantic modeling, often achieved through conceptual networks or graphs. These tools enable the structuring of knowledge within a domain by accounting for relationships between concepts, contexts of use, and specific objectives. Stockinger also highlights the constraints and challenges involved in creating dynamic and adaptable models, integrating elements such as thesauri or interoperable ontologies to facilitate the analysis and publication of complex corpora.
3) Applications and Multimodal Visualization:  The text concludes by examining the practical application of these models in work environments like OKAPI, developed to analyze, publish, and reuse audiovisual data. It also discusses innovative approaches such as visual storytelling and document reengineering, which involve transforming existing content into new resources tailored to various contexts. These methods emphasize interoperability, flexibility, and the intelligence of communication systems, paving the way for richer and more collaborative use of digital data. The content of this document was presented during the "Semiotics of Information Design" Day organized by Anne Beyaert-Geslin of the University of Bordeaux Montaigne (MICA laboratory) on June 21, 2018, in Bordeaux. 

**Abstract (ZH)**: 1) 介绍与概念框架：本文探讨了信息设计的概念，将其分为两大实践：定义文本数据集的意义及其视觉或多媒体表示。本文借鉴了丰富文本语料库，尤其是音视频语料库方面的专业知识，并将其转换为多种叙事格式。文章突出了领域语义内容与图形表达的模式之间的关键区别，通过结构语义学和语言学传统中的概念来说明这一方法。

2) 模型与概念设计：文章强调了语义建模的重要性，通常通过概念网络或图来实现。这些工具能够通过考虑概念间的关联、使用场合以及特定目标来在领域内结构化知识。Stockinger还强调了在创建动态和可适应模型过程中所面临的约束与挑战，通过整合如词汇表或互操作本体等元素，来促进复杂语料库的分析与发布。

3) 应用与多媒体可视化：本文最后考察了这些模型在OKAPI等工作环境中的实际应用，OKAPI开发用于分析、发布和重用音视频数据。此外，文章还讨论了视觉叙事和文档重构等创新方法，涉及将现有内容转化为适应不同上下文的新资源。这些方法强调了互操作性、灵活性和通信系统的智能性，为更丰富和更具协作性的数字数据使用铺平了道路。本文内容于2018年6月21日在波尔多举行的由波尔多蒙吐内大学Anne Beyaert-Geslin组织的“信息设计语义学”日活动中呈现，该活动由MICA实验室主办。 

---
# Towards Explainable Fusion and Balanced Learning in Multimodal Sentiment Analysis 

**Title (ZH)**: 面向可解释融合与平衡学习的多模态情感分析 

**Authors**: Miaosen Luo, Yuncheng Jiang, Sijie Mai  

**Link**: [PDF](https://arxiv.org/pdf/2504.12151)  

**Abstract**: Multimodal Sentiment Analysis (MSA) faces two critical challenges: the lack of interpretability in the decision logic of multimodal fusion and modality imbalance caused by disparities in inter-modal information density. To address these issues, we propose KAN-MCP, a novel framework that integrates the interpretability of Kolmogorov-Arnold Networks (KAN) with the robustness of the Multimodal Clean Pareto (MCPareto) framework. First, KAN leverages its univariate function decomposition to achieve transparent analysis of cross-modal interactions. This structural design allows direct inspection of feature transformations without relying on external interpretation tools, thereby ensuring both high expressiveness and interpretability. Second, the proposed MCPareto enhances robustness by addressing modality imbalance and noise interference. Specifically, we introduce the Dimensionality Reduction and Denoising Modal Information Bottleneck (DRD-MIB) method, which jointly denoises and reduces feature dimensionality. This approach provides KAN with discriminative low-dimensional inputs to reduce the modeling complexity of KAN while preserving critical sentiment-related information. Furthermore, MCPareto dynamically balances gradient contributions across modalities using the purified features output by DRD-MIB, ensuring lossless transmission of auxiliary signals and effectively alleviating modality imbalance. This synergy of interpretability and robustness not only achieves superior performance on benchmark datasets such as CMU-MOSI, CMU-MOSEI, and CH-SIMS v2 but also offers an intuitive visualization interface through KAN's interpretable architecture. 

**Abstract (ZH)**: 多模态情感分析中的KAN-MCP框架：可解释性和鲁棒性的协同提升 

---
# Towards Safe Synthetic Image Generation On the Web: A Multimodal Robust NSFW Defense and Million Scale Dataset 

**Title (ZH)**: 面向网络的合成图像安全生成：多模态鲁棒不适图防护及千万规模数据集 

**Authors**: Muhammad Shahid Muneer, Simon S. Woo  

**Link**: [PDF](https://arxiv.org/pdf/2504.11707)  

**Abstract**: In the past years, we have witnessed the remarkable success of Text-to-Image (T2I) models and their widespread use on the web. Extensive research in making T2I models produce hyper-realistic images has led to new concerns, such as generating Not-Safe-For-Work (NSFW) web content and polluting the web society. To help prevent misuse of T2I models and create a safer web environment for users features like NSFW filters and post-hoc security checks are used in these models. However, recent work unveiled how these methods can easily fail to prevent misuse. In particular, adversarial attacks on text and image modalities can easily outplay defensive measures. %Exploiting such leads to the growing concern of preventing adversarial attacks on text and image modalities. Moreover, there is currently no robust multimodal NSFW dataset that includes both prompt and image pairs and adversarial examples. This work proposes a million-scale prompt and image dataset generated using open-source diffusion models. Second, we develop a multimodal defense to distinguish safe and NSFW text and images, which is robust against adversarial attacks and directly alleviates current challenges. Our extensive experiments show that our model performs well against existing SOTA NSFW detection methods in terms of accuracy and recall, drastically reducing the Attack Success Rate (ASR) in multimodal adversarial attack scenarios. Code: this https URL. 

**Abstract (ZH)**: 近年来，我们见证了文本到图像（T2I）模型的显著成功及其在网上的广泛应用。为使T2I模型生成超现实图像的广泛研究已引发新的担忧，如生成不适合工作（NSFW）的网络内容和污染网络社会。为帮助防止T2I模型的滥用并为用户提供更安全的网络环境，这些模型中使用了像NSFW滤镜和事后安全检查这样的功能。然而，近期研究表明，这些方法容易失效。特别是，对文本和图像模态的对抗攻击可以轻易地超越防御措施。利用这些攻击的方法引发了防止对文本和图像模态进行对抗攻击的关注。此外，目前尚无包含提示和图像对以及对抗示例的稳健多模态NSFW数据集。本工作提出一个基于开源扩散模型生成的百万规模提示和图像数据集。其次，我们开发了一种多模态防御方法，用于区分安全和NSFW的文本和图像，并且对对抗攻击表现出鲁棒性，直接缓解了当前的挑战。我们的实验表明，与现有的顶级NSFW检测方法相比，在准确率和召回率方面，我们的模型表现出色，大幅降低了多模态对抗攻击场景中的攻击成功率（ASR）。代码：![](this https URL) 

---
# Can GPT tell us why these images are synthesized? Empowering Multimodal Large Language Models for Forensics 

**Title (ZH)**: GPT能告诉我们这些图像为何是合成的吗？赋能多模态大型语言模型在 forensic 领域的应用 

**Authors**: Yiran He, Yun Cao, Bowen Yang, Zeyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.11686)  

**Abstract**: The rapid development of generative AI facilitates content creation and makes image manipulation easier and more difficult to detect. While multimodal Large Language Models (LLMs) have encoded rich world knowledge, they are not inherently tailored for combating AI-generated Content (AIGC) and struggle to comprehend local forgery details. In this work, we investigate the application of multimodal LLMs in forgery detection. We propose a framework capable of evaluating image authenticity, localizing tampered regions, providing evidence, and tracing generation methods based on semantic tampering clues. Our method demonstrates that the potential of LLMs in forgery analysis can be effectively unlocked through meticulous prompt engineering and the application of few-shot learning techniques. We conduct qualitative and quantitative experiments and show that GPT4V can achieve an accuracy of 92.1% in Autosplice and 86.3% in LaMa, which is competitive with state-of-the-art AIGC detection methods. We further discuss the limitations of multimodal LLMs in such tasks and propose potential improvements. 

**Abstract (ZH)**: 生成式AI的快速发展促进了内容创作并使得图像操纵更加容易且更难以检测。尽管多模态大型语言模型（LLM）已嵌入丰富的世界知识，但它们并非天生适合对抗AI生成内容（AIGC），难以理解局部篡改细节。本文探讨了多模态LLM在伪造检测中的应用。我们提出了一种框架，能够评估图像的真实性、定位篡改区域、提供证据并基于语义篡改线索追踪生成方法。我们的方法表明，通过细致的提示工程和少样本学习技术的应用，可以有效解锁LLM在伪造分析中的潜在能力。我们进行了定性和定量实验，表明GPT4V在Autosplice中的准确率为92.1%，在LaMa中的准确率为86.3%，其性能与最先进的AIGC检测方法相当。我们进一步讨论了多模态LLM在这些任务中的局限性，并提出了潜在的改进方案。 

---
# Visual moral inference and communication 

**Title (ZH)**: 视觉道德推理与沟通 

**Authors**: Warren Zhu, Aida Ramezani, Yang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2504.11473)  

**Abstract**: Humans can make moral inferences from multiple sources of input. In contrast, automated moral inference in artificial intelligence typically relies on language models with textual input. However, morality is conveyed through modalities beyond language. We present a computational framework that supports moral inference from natural images, demonstrated in two related tasks: 1) inferring human moral judgment toward visual images and 2) analyzing patterns in moral content communicated via images from public news. We find that models based on text alone cannot capture the fine-grained human moral judgment toward visual stimuli, but language-vision fusion models offer better precision in visual moral inference. Furthermore, applications of our framework to news data reveal implicit biases in news categories and geopolitical discussions. Our work creates avenues for automating visual moral inference and discovering patterns of visual moral communication in public media. 

**Abstract (ZH)**: 人类可以从多种输入来源进行道德推理，而人工智能的自动道德推理通常依赖于基于文本的语言模型。然而，道德还通过语言之外的其他模态传达。我们提出了一种计算框架，支持从自然图像进行道德推理，并通过两个相关任务进行了演示：1) 推断人类对视觉图像的道德判断；2) 分析通过公共新闻传播的道德内容模式。我们发现仅基于文本的模型无法捕捉人类对视觉刺激的细微道德判断，但语言-视觉融合模型在视觉道德推理方面提供了更高的精度。此外，将我们的框架应用于新闻数据揭示了新闻类别和地缘政治讨论中的隐含偏见。我们的研究为自动化视觉道德推理和发现公共媒体中视觉道德沟通模式提供了途径。 

---
# Semantic Matters: Multimodal Features for Affective Analysis 

**Title (ZH)**: 语义为本：多模态特征的情感分析 

**Authors**: Tobias Hallmen, Robin-Nico Kampa, Fabian Deuser, Norbert Oswald, Elisabeth André  

**Link**: [PDF](https://arxiv.org/pdf/2504.11460)  

**Abstract**: In this study, we present our methodology for two tasks: the Behavioural Ambivalence/Hesitancy (BAH) Recognition Challenge and the Emotional Mimicry Intensity (EMI) Estimation Challenge, both conducted as part of the 8th Workshop and Competition on Affective & Behavior Analysis in-the-wild. Building on previous work, we utilize a Wav2Vec 2.0 model pre-trained on a large podcast dataset to extract various audio features, capturing both linguistic and paralinguistic information. Our approach incorporates a valence-arousal-dominance (VAD) module derived from Wav2Vec 2.0, a BERT-like encoder, and a vision transformer (ViT) with predictions subsequently processed through a long short-term memory (LSTM) architecture for temporal modeling. In this iteration, we integrate the textual and visual modality into our analysis, recognizing that semantic content provides valuable contextual cues and underscoring that the meaning of speech often conveys more critical insights than its acoustic counterpart alone. Fusing in the vision modality helps in some cases to interpret the textual modality more precisely. This combined approach yields significant performance improvements over baseline methods. 

**Abstract (ZH)**: 本文研究了在第8届野生情感与行为分析研讨会与竞赛中的两项任务：行为 ambivalence/犹豫（BAH）识别挑战和情感模仿强度（EMI）估计挑战的方案。我们构建的方法基于预训练于大量播客数据集的Wav2Vec 2.0模型提取多种音频特征，涵盖语言和副语言信息。该方法包含源自Wav2Vec 2.0的情感极性-唤醒-支配（VAD）模块、类似BERT的编码器和视觉变换器（ViT），预测结果随后通过长短期记忆（LSTM）架构进行时间建模。本研究将文本和视觉模态整合到分析中，强调语义内容提供的上下文线索价值，指出言语的意义往往比其声学特征本身提供更多的关键见解。在某些情况下，整合视觉模态有助于更精确地解释文本模态。这种结合的方法相较于基线方法取得了显著的性能提升。 

---
