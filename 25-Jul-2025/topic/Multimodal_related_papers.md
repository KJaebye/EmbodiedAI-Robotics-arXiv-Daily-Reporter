# OpenNav: Open-World Navigation with Multimodal Large Language Models 

**Title (ZH)**: OpenNav: 多模态大型语言模型支持的开放世界导航 

**Authors**: Mingfeng Yuan, Letian Wang, Steven L. Waslander  

**Link**: [PDF](https://arxiv.org/pdf/2507.18033)  

**Abstract**: Pre-trained large language models (LLMs) have demonstrated strong common-sense reasoning abilities, making them promising for robotic navigation and planning tasks. However, despite recent progress, bridging the gap between language descriptions and actual robot actions in the open-world, beyond merely invoking limited predefined motion primitives, remains an open challenge. In this work, we aim to enable robots to interpret and decompose complex language instructions, ultimately synthesizing a sequence of trajectory points to complete diverse navigation tasks given open-set instructions and open-set objects. We observe that multi-modal large language models (MLLMs) exhibit strong cross-modal understanding when processing free-form language instructions, demonstrating robust scene comprehension. More importantly, leveraging their code-generation capability, MLLMs can interact with vision-language perception models to generate compositional 2D bird-eye-view value maps, effectively integrating semantic knowledge from MLLMs with spatial information from maps to reinforce the robot's spatial understanding. To further validate our approach, we effectively leverage large-scale autonomous vehicle datasets (AVDs) to validate our proposed zero-shot vision-language navigation framework in outdoor navigation tasks, demonstrating its capability to execute a diverse range of free-form natural language navigation instructions while maintaining robustness against object detection errors and linguistic ambiguities. Furthermore, we validate our system on a Husky robot in both indoor and outdoor scenes, demonstrating its real-world robustness and applicability. Supplementary videos are available at this https URL 

**Abstract (ZH)**: 预训练大型语言模型（LLMs）展现了强大的常识推理能力，使其在机器人导航和规划任务中具有潜力。然而，尽管取得了近期进展，如何在开放世界中弥合语言描述与实际机器人动作之间的差距，而不仅仅是调用有限的预定义运动 primitive，仍然是一个开放的挑战。在本文中，我们旨在使机器人能够解释和分解复杂的语言指令，最终在给定开放指令和开放对象的情况下合成轨迹点序列，以完成各种导航任务。我们观察到，多模态大型语言模型（MLLMs）在处理自由形式的语言指令时表现出强大的跨模态理解能力，展示了稳健的场景理解。更重要的是，利用其代码生成能力，MLLMs 可以与视觉-语言感知模型互动，生成组合的2D 鸟瞰图价值图，有效地将MLLMs 的语义知识与地图中的空间信息整合，增强机器人的空间理解。为了进一步验证我们提出的方法，我们有效地利用大规模的自主车辆数据集（AVDs）来验证我们提出的零样本视觉-语言导航框架在室外导航任务中的有效性，展示了其在执行多样化自由形式自然语言导航指令的同时，保持着对物体检测错误和语义歧义的鲁棒性。此外，我们在 Husky 机器人上验证了我们的系统，无论是在室内还是室外场景，都展示了其实用性和鲁棒性。相关的补充视频可在此网址查看。 

---
# VideoMind: An Omni-Modal Video Dataset with Intent Grounding for Deep-Cognitive Video Understanding 

**Title (ZH)**: VideoMind: 一种具有意图 grounding 的全模态视频数据集，用于深度认知视频理解 

**Authors**: Baoyao Yang, Wanyun Li, Dixin Chen, Junxiang Chen, Wenbin Yao, Haifeng Lin  

**Link**: [PDF](https://arxiv.org/pdf/2507.18552)  

**Abstract**: This paper introduces VideoMind, a video-centric omni-modal dataset designed for deep video content cognition and enhanced multi-modal feature representation. The dataset comprises 103K video samples (3K reserved for testing), each paired with audio and systematically detailed textual descriptions. Specifically, every video and its audio is described across three hierarchical layers (factual, abstract, and intent), progressing from surface to depth. It contains over 22 million words, averaging ~225 words per sample. VideoMind's key distinction from existing datasets is its provision of intent expressions, which require contextual integration across the entire video and are not directly observable. These deep-cognitive expressions are generated using a Chain-of-Thought (COT) approach, prompting the mLLM through step-by-step reasoning. Each description includes annotations for subject, place, time, event, action, and intent, supporting downstream recognition tasks. Crucially, we establish a gold-standard benchmark with 3,000 manually validated samples for evaluating deep-cognitive video understanding. We design hybrid-cognitive retrieval experiments, scored by multi-level retrieval metrics, to appropriately assess deep video comprehension. Evaluation results for models (e.g., InternVideo, VAST, UMT-L) are released. VideoMind serves as a powerful benchmark for fine-grained cross-modal alignment and advances fields requiring in-depth video understanding, such as emotion and intent recognition. The data is publicly available on GitHub, HuggingFace, and OpenDataLab, this https URL. 

**Abstract (ZH)**: VideoMind：一种面向视频的全模态数据集，用于深度视频内容认知和增强多模态特征表示 

---
# Explaining How Visual, Textual and Multimodal Encoders Share Concepts 

**Title (ZH)**: 解释视觉、文本和多模态编码器如何共享概念 

**Authors**: Clément Cornet, Romaric Besançon, Hervé Le Borgne  

**Link**: [PDF](https://arxiv.org/pdf/2507.18512)  

**Abstract**: Sparse autoencoders (SAEs) have emerged as a powerful technique for extracting human-interpretable features from neural networks activations. Previous works compared different models based on SAE-derived features but those comparisons have been restricted to models within the same modality. We propose a novel indicator allowing quantitative comparison of models across SAE features, and use it to conduct a comparative study of visual, textual and multimodal encoders. We also propose to quantify the Comparative Sharedness of individual features between different classes of models. With these two new tools, we conduct several studies on 21 encoders of the three types, with two significantly different sizes, and considering generalist and domain specific datasets. The results allow to revisit previous studies at the light of encoders trained in a multimodal context and to quantify to which extent all these models share some representations or features. They also suggest that visual features that are specific to VLMs among vision encoders are shared with text encoders, highlighting the impact of text pretraining. The code is available at this https URL 

**Abstract (ZH)**: 稀疏自编码器（SAEs）已 emerge 作为从神经网络激活中提取人类可解读特征的强而有力的技术。以往的研究基于 SAE 提取的特征比较了不同的模型，但这些比较局限于同一模态的模型之内。我们提出了一种新的指标，用于跨 SAE 特征定量比较模型，并使用该指标对视觉、文本和多模态编码器进行了比较研究。我们还提出了一种量化不同模型类别之间单个特征对比共享性的方法。利用这两种新的工具，我们对三类共 21 个编码器进行了研究，这些编码器具有两种显着不同的规模，并考虑了通用和特定领域的数据集。研究结果允许我们在多模态训练的编码器背景下重新审视之前的研究所提出的观点，并定量分析这些模型在多大程度上共享一些表示或特征。研究还表明，视觉编码器中特定于 VLM 的特征与文本编码器共享，突显了文本预训练的影响。代码可在以下网址获取：this https URL。 

---
# Multimodal Behavioral Patterns Analysis with Eye-Tracking and LLM-Based Reasoning 

**Title (ZH)**: 基于眼动追踪和大语言模型驱动的推理的多模态行为模式分析 

**Authors**: Dongyang Guo, Yasmeen Abdrabou, Enkeleda Thaqi, Enkelejda Kasneci  

**Link**: [PDF](https://arxiv.org/pdf/2507.18252)  

**Abstract**: Eye-tracking data reveals valuable insights into users' cognitive states but is difficult to analyze due to its structured, non-linguistic nature. While large language models (LLMs) excel at reasoning over text, they struggle with temporal and numerical data. This paper presents a multimodal human-AI collaborative framework designed to enhance cognitive pattern extraction from eye-tracking signals. The framework includes: (1) a multi-stage pipeline using horizontal and vertical segmentation alongside LLM reasoning to uncover latent gaze patterns; (2) an Expert-Model Co-Scoring Module that integrates expert judgment with LLM output to generate trust scores for behavioral interpretations; and (3) a hybrid anomaly detection module combining LSTM-based temporal modeling with LLM-driven semantic analysis. Our results across several LLMs and prompt strategies show improvements in consistency, interpretability, and performance, with up to 50% accuracy in difficulty prediction tasks. This approach offers a scalable, interpretable solution for cognitive modeling and has broad potential in adaptive learning, human-computer interaction, and educational analytics. 

**Abstract (ZH)**: 眼动数据揭示了用户认知状态的重要见解，但由于其结构化且非语言的性质，分析起来非常困难。虽然大型语言模型在处理文本方面表现出色，但在处理时间和数值数据方面却存在困难。本文提出了一种多模态人机协作框架，旨在增强从眼动信号中提取认知模式的能力。该框架包括：（1）多阶段管道，结合水平和垂直分割以及大型语言模型推理，以揭示潜在的眼动模式；（2）专家模型协作评分模块，结合专家判断与大型语言模型输出生成行为解释的信任分；以及（3）结合基于LSTM的时间建模与大型语言模型驱动的语义分析的混合异常检测模块。我们的实验结果表明，该方法在不同大型语言模型和提示策略下提高了一致性和可解释性，并在难度预测任务中最高可达50%的准确性。该方法提供了一种可扩展且可解释的认知建模解决方案，并在自适应学习、人机交互和教育分析等领域具有广泛潜力。 

---
# Bob's Confetti: Phonetic Memorization Attacks in Music and Video Generation 

**Title (ZH)**: 鲍勃的彩纸：音乐和视频生成中的phonetic记忆攻击 

**Authors**: Jaechul Roh, Zachary Novack, Yuefeng Peng, Niloofar Mireshghallah, Taylor Berg-Kirkpatrick, Amir Houmansadr  

**Link**: [PDF](https://arxiv.org/pdf/2507.17937)  

**Abstract**: Lyrics-to-Song (LS2) generation models promise end-to-end music synthesis from text, yet their vulnerability to training data memorization remains underexplored. We introduce Adversarial PhoneTic Prompting (APT), a novel attack where lyrics are semantically altered while preserving their acoustic structure through homophonic substitutions (e.g., Eminem's famous "mom's spaghetti" $\rightarrow$ "Bob's confetti"). Despite these distortions, we uncover a powerful form of sub-lexical memorization: models like SUNO and YuE regenerate outputs strikingly similar to known training content, achieving high similarity across audio-domain metrics, including CLAP, AudioJudge, and CoverID. This vulnerability persists across multiple languages and genres. More surprisingly, we discover that phoneme-altered lyrics alone can trigger visual memorization in text-to-video models. When prompted with phonetically modified lyrics from Lose Yourself, Veo 3 reconstructs visual elements from the original music video -- including character appearance and scene composition -- despite no visual cues in the prompt. We term this phenomenon phonetic-to-visual regurgitation. Together, these findings expose a critical vulnerability in transcript-conditioned multimodal generation: phonetic prompting alone can unlock memorized audiovisual content, raising urgent questions about copyright, safety, and content provenance in modern generative systems. Example generations are available on our demo page (this http URL). 

**Abstract (ZH)**: Lyrics-to-Song (LS2)生成模型 promise 从文本到端音乐合成，但其在训练数据记忆化方面的脆弱性仍需进一步探索。我们介绍了对抗音素提示 (Adversarial PhoneTic Prompting, APT)，这是一种新颖的攻击方式，通过同音替换改变歌词的语义同时保持其声学结构（例如， Eminem 著名的 "mom's spaghetti" $\rightarrow$ "Bob's confetti"）。尽管存在这些扭曲，我们发现了一种强大的次词层记忆形式：如 SUNO 和 YuE 这样的模型会生成与已知训练内容惊人相似的输出，这些输出在包括 CLAP、AudioJudge 和 CoverID 的音频域度量标准上实现了高相似度。这种脆弱性跨越了多种语言和流派。更为令人惊讶的是，我们发现仅仅通过音素修改的歌词就可以触发文本到视频模型中的视觉记忆。当使用《Lose Yourself》中音素修改的歌词提示时，Veo 3 重构了原始音乐视频中的视觉元素，包括角色外观和场景组成——尽管提示中没有任何视觉线索。我们将这一现象称为音素到视觉的回吐。这些发现揭示了转录条件下的多模态生成中一个关键的脆弱性：仅通过音素提示就可以解锁记忆中的音频-视觉内容，这引发了关于现代生成系统中版权、安全性和内容来源的紧急问题。示例生成可在我们的演示页面上获得。 

---
