# Visual Graph Arena: Evaluating Visual Conceptualization of Vision and Multimodal Large Language Models 

**Title (ZH)**: 视觉图腾 arena：评估视觉概念化能力的视觉和多模态大型语言模型 

**Authors**: Zahra Babaiee, Peyman M. Kiasari, Daniela Rus, Radu Grosu  

**Link**: [PDF](https://arxiv.org/pdf/2506.06242)  

**Abstract**: Recent advancements in multimodal large language models have driven breakthroughs in visual question answering. Yet, a critical gap persists, `conceptualization'-the ability to recognize and reason about the same concept despite variations in visual form, a basic ability of human reasoning. To address this challenge, we introduce the Visual Graph Arena (VGA), a dataset featuring six graph-based tasks designed to evaluate and improve AI systems' capacity for visual abstraction. VGA uses diverse graph layouts (e.g., Kamada-Kawai vs. planar) to test reasoning independent of visual form. Experiments with state-of-the-art vision models and multimodal LLMs reveal a striking divide: humans achieved near-perfect accuracy across tasks, while models totally failed on isomorphism detection and showed limited success in path/cycle tasks. We further identify behavioral anomalies suggesting pseudo-intelligent pattern matching rather than genuine understanding. These findings underscore fundamental limitations in current AI models for visual understanding. By isolating the challenge of representation-invariant reasoning, the VGA provides a framework to drive progress toward human-like conceptualization in AI visual models. The Visual Graph Arena is available at: \href{this https URL}{this http URL} 

**Abstract (ZH)**: 近期多模态大型语言模型的进展推动了视觉问答领域的突破。然而，仍存在一个关键缺口，即“概念化”——识别和推理同一概念的能力，尽管其视觉形式存在差异，这是人类推理的基本能力。为应对这一挑战，我们提出了视觉图场（VGA），一个包含六项基于图的任务的数据集，旨在评估和提高AI系统在视觉抽象方面的能力。VGA使用多样化的图布局（例如，Kamada-Kawai vs. 平面布局）来测试独立于视觉形式的推理能力。使用最先进的视觉模型和多模态大语言模型的实验揭示了一个明显的鸿沟：人类在所有任务中几乎实现了完美的准确性，而模型在同构性检测上完全失败，并且在路径/环路任务上表现出有限的成功。我们进一步识别了行为异常，表明伪智能的模式匹配而不是真正的理解。这些发现强调了当前AI模型在视觉理解方面的根本局限性。通过将表示不变的推理挑战隔离出来，VGA提供了一个框架，以促进开发出类似人类的概念化能力的AI视觉模型。视觉图场数据集可在以下链接获取：\href{this https URL}{this http URL}。 

---
# PuzzleWorld: A Benchmark for Multimodal, Open-Ended Reasoning in Puzzlehunts 

**Title (ZH)**: PuzzleWorld: 一款用于谜题 hunt 多模态开放性推理的基准测试 

**Authors**: Hengzhi Li, Brendon Jiang, Alexander Naehu, Regan Song, Justin Zhang, Megan Tjandrasuwita, Chanakya Ekbote, Steven-Shine Chen, Adithya Balachandran, Wei Dai, Rebecca Chang, Paul Pu Liang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06211)  

**Abstract**: Puzzlehunts are a genre of complex, multi-step puzzles lacking well-defined problem definitions. In contrast to conventional reasoning benchmarks consisting of tasks with clear instructions, puzzlehunts require models to discover the underlying problem structure from multimodal evidence and iterative reasoning, mirroring real-world domains such as scientific discovery, exploratory data analysis, or investigative problem-solving. Despite recent progress in foundation models, their performance on such open-ended settings remains largely untested. In this paper, we introduce PuzzleWorld, a large-scale benchmark of 667 puzzlehunt-style problems designed to assess step-by-step, open-ended, and creative multimodal reasoning. Each puzzle is annotated with the final solution, detailed reasoning traces, and cognitive skill labels, enabling holistic benchmarking and fine-grained diagnostic analysis. Most state-of-the-art models achieve only 1-2% final answer accuracy, with the best model solving only 14% of puzzles and reaching 40% stepwise accuracy. To demonstrate the value of our reasoning annotations, we show that fine-tuning a small model on reasoning traces improves stepwise reasoning from 4% to 11%, while training on final answers alone degrades performance to near zero. Our error analysis reveals that current models exhibit myopic reasoning, are bottlenecked by the limitations of language-based inference, and lack sketching capabilities crucial for visual and spatial reasoning. We release PuzzleWorld at this https URL to support future work on building more general, open-ended, and creative reasoning systems. 

**Abstract (ZH)**: 谜题搜寻是一种缺乏明确问题定义的复杂多步谜题类型。与传统推理基准中任务明确指示的模式不同，谜题搜寻要求模型从多模态证据和迭代推理中发现潜在的问题结构，这与科学发现、探索性数据分析或调查性问题解决等现实世界领域相映射。尽管基础模型的进展取得了进步，但在这种开放式设置中的性能仍然很少被测试。本文我们引入了PuzzleWorld，这是一个包含667个谜题搜寻样问题的大规模基准，旨在评估逐步、开放式和创造性的多模态推理能力。每个谜题都带有最终解决方案、详细的推理轨迹和认知技能标签，支持全面基准测试和细粒度诊断分析。大多数最先进的模型仅达到1-2%的最终答案准确性，最佳模型也只能解决14%的谜题，并且逐步准确性达到40%。为了展示我们推理注释的价值，我们展示了在推理轨迹上微调一个小模型能将逐步推理从4%提升到11%，而仅使用最终答案进行训练则会将性能降低到接近零。我们的错误分析揭示了当前模型表现出短视推理、受限于基于语言的推理的局限性，并且缺乏对于视觉和空间推理至关重要的绘图能力。我们在此提供PuzzleWorld以支持未来构建更通用、开放式和创造性推理系统的研究。 

---
# WhisQ: Cross-Modal Representation Learning for Text-to-Music MOS Prediction 

**Title (ZH)**: WhisQ: 跨模态表示学习用于文本到音乐主观质量预测 

**Authors**: Jakaria Islam Emon, Kazi Tamanna Alam, Md. Abu Salek  

**Link**: [PDF](https://arxiv.org/pdf/2506.05899)  

**Abstract**: Mean Opinion Score (MOS) prediction for text to music systems requires evaluating both overall musical quality and text prompt alignment. This paper introduces WhisQ, a multimodal architecture that addresses this dual-assessment challenge through sequence level co-attention and optimal transport regularization. WhisQ employs the Whisper Base pretrained model for temporal audio encoding and Qwen 3, a 0.6B Small Language Model (SLM), for text encoding, with both maintaining sequence structure for fine grained cross-modal modeling. The architecture features specialized prediction pathways: OMQ is predicted from pooled audio embeddings, while TA leverages bidirectional sequence co-attention between audio and text. Sinkhorn optimal transport loss further enforce semantic alignment in the shared embedding space. On the MusicEval Track-1 dataset, WhisQ achieves substantial improvements over the baseline: 7% improvement in Spearman correlation for OMQ and 14% for TA. Ablation studies reveal that optimal transport regularization provides the largest performance gain (10% SRCC improvement), demonstrating the importance of explicit cross-modal alignment for text-to-music evaluation. 

**Abstract (ZH)**: 基于文本到音乐系统的Mean Opinion Score (MOS)预测需要评估整体音乐质量和文本提示对齐。本文介绍了WhisQ，这是一种多模态架构，通过序列级别共注意力和最优传输正则化来应对这种双重评估挑战。WhisQ 使用 Whisper Base 预训练模型进行时序音频编码，并使用 Qwen 3（一个 0.6B 小型语言模型）进行文本编码，两者都保持了序列结构以进行精细粒度的跨模态建模。该架构具有专门的预测路径：OMQ 从聚合的音频嵌入中预测，而 TA 则利用音频与文本之间的双向序列共注意力。最优传输损失进一步在共享嵌入空间中强制执行语义对齐。在 MusicEval Track-1 数据集上，WhisQ 在 OMQ 和 TA 方面均实现了显著提升：OMQ 的 Spearman 相关系数提高了 7%，TA 提高了 14%。消融研究结果显示，最优传输正则化提供了最大的性能增益（Spearman 相关系数提高了 10%），这表明明确的跨模态对齐对于文本到音乐评估的重要性。 

---
# Cross-View Multi-Modal Segmentation @ Ego-Exo4D Challenges 2025 

**Title (ZH)**: Ego-Exo4D 挑战赛 2025 多模态跨视角分割 

**Authors**: Yuqian Fu, Runze Wang, Yanwei Fu, Danda Pani Paudel, Luc Van Gool  

**Link**: [PDF](https://arxiv.org/pdf/2506.05856)  

**Abstract**: In this report, we present a cross-view multi-modal object segmentation approach for the object correspondence task in the Ego-Exo4D Correspondence Challenges 2025. Given object queries from one perspective (e.g., ego view), the goal is to predict the corresponding object masks in another perspective (e.g., exo view). To tackle this task, we propose a multimodal condition fusion module that enhances object localization by leveraging both visual masks and textual descriptions as segmentation conditions. Furthermore, to address the visual domain gap between ego and exo views, we introduce a cross-view object alignment module that enforces object-level consistency across perspectives, thereby improving the model's robustness to viewpoint changes. Our proposed method ranked second on the leaderboard of the large-scale Ego-Exo4D object correspondence benchmark. Code will be made available at this https URL. 

**Abstract (ZH)**: 在本报告中，我们提出了一个用于Ego-Exo4D对应挑战2025中的对象对应任务的跨视角多模态对象分割方法。给定一个视角的对象查询（例如，第一人称视图），目标是在另一个视角（例如，第三人称视图）中预测相应的对象掩码。为了解决这一任务，我们提出了一种多模态条件融合模块，通过利用视觉掩码和文本描述作为分割条件来增强对象定位。此外，为了应对第一人称视图和第三人称视图之间的视觉域差距，我们引入了一种跨视角对象对齐模块，确保各视角之间的一致性，从而提高模型对视角变化的鲁棒性。我们的提出的方法在大型Ego-Exo4D对象对应基准测试的排行榜上排名第二。代码将在以下链接处提供：这个 https URL。 

---
# DeepFake Doctor: Diagnosing and Treating Audio-Video Fake Detection 

**Title (ZH)**: DeepFake医生：音频-视频伪造检测的诊断与治疗 

**Authors**: Marcel Klemt, Carlotta Segna, Anna Rohrbach  

**Link**: [PDF](https://arxiv.org/pdf/2506.05851)  

**Abstract**: Generative AI advances rapidly, allowing the creation of very realistic manipulated video and audio. This progress presents a significant security and ethical threat, as malicious users can exploit DeepFake techniques to spread misinformation. Recent DeepFake detection approaches explore the multimodal (audio-video) threat scenario. In particular, there is a lack of reproducibility and critical issues with existing datasets - such as the recently uncovered silence shortcut in the widely used FakeAVCeleb dataset. Considering the importance of this topic, we aim to gain a deeper understanding of the key issues affecting benchmarking in audio-video DeepFake detection. We examine these challenges through the lens of the three core benchmarking pillars: datasets, detection methods, and evaluation protocols. To address these issues, we spotlight the recent DeepSpeak v1 dataset and are the first to propose an evaluation protocol and benchmark it using SOTA models. We introduce SImple Multimodal BAseline (SIMBA), a competitive yet minimalistic approach that enables the exploration of diverse design choices. We also deepen insights into the issue of audio shortcuts and present a promising mitigation strategy. Finally, we analyze and enhance the evaluation scheme on the widely used FakeAVCeleb dataset. Our findings offer a way forward in the complex area of audio-video DeepFake detection. 

**Abstract (ZH)**: Generative AI在音频视频DeepFake检测基准测试中的挑战与对策：从数据集、检测方法和评估协议三个核心方面深入探究 

---
# MORSE-500: A Programmatically Controllable Video Benchmark to Stress-Test Multimodal Reasoning 

**Title (ZH)**: MORSE-500：一个可编程控制的视频基准测试，用于压力测试多模态推理能力 

**Authors**: Zikui Cai, Andrew Wang, Anirudh Satheesh, Ankit Nakhawa, Hyunwoo Jae, Keenan Powell, Minghui Liu, Neel Jay, Sungbin Oh, Xiyao Wang, Yongyuan Liang, Tom Goldstein, Furong Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05523)  

**Abstract**: Despite rapid advances in vision-language models (VLMs), current benchmarks for multimodal reasoning fall short in three key dimensions. First, they overwhelmingly rely on static images, failing to capture the temporal complexity of real-world environments. Second, they narrowly focus on mathematical problem-solving, neglecting the broader spectrum of reasoning skills -- including abstract, physical, planning, spatial, and temporal capabilities -- required for robust multimodal intelligence. Third, many benchmarks quickly saturate, offering limited headroom for diagnosing failure modes or measuring continued progress. We introduce MORSE-500 (Multimodal Reasoning Stress-test Environment), a video benchmark composed of 500 fully scripted clips with embedded questions spanning six complementary reasoning categories. Each instance is programmatically generated using deterministic Python scripts (via Manim, Matplotlib, MoviePy), generative video models, and curated real footage. This script-driven design allows fine-grained control over visual complexity, distractor density, and temporal dynamics -- enabling difficulty to be scaled systematically as models improve. Unlike static benchmarks that become obsolete once saturated, MORSE-500 is built to evolve: its controllable generation pipeline supports the creation of arbitrarily challenging new instances, making it ideally suited for stress-testing next-generation models. Initial experiments with state-of-the-art systems -- including various Gemini 2.5 Pro and OpenAI o3 which represent the strongest available at the time, alongside strong open-source models -- reveal substantial performance gaps across all categories, with particularly large deficits in abstract and planning tasks. We release the full dataset, generation scripts, and evaluation harness to support transparent, reproducible, and forward-looking multimodal reasoning research. 

**Abstract (ZH)**: 尽管视觉语言模型取得了快速进步，当前的多模态推理基准在三个关键维度上仍存在不足。首先，它们主要依赖静态图像，未能捕捉到现实世界环境的时间复杂性。其次，它们狭隘地集中在数学问题解决上，忽视了实现稳健的多模态智能所需更广泛的推理技能范围，包括抽象、物理、规划、空间和时间能力。第三，许多基准很快达到饱和，提供有限的空间来诊断失败模式或衡量持续进步。我们引入了MORSE-500（多模态推理压力测试环境），这是一个由500个完全脚本化的视频片段组成的数据集，涵盖六个互补的推理类别，并嵌入了问题。每一实例都是通过确定性的Python脚本（使用Manim、Matplotlib、MoviePy）、生成性视频模型和精选的真实素材程序化生成的。这种脚本驱动的设计允许对视觉复杂性、干扰密度和时间动态性的精确控制——从而使难度可以随着模型的进步而系统地调整。不同于一旦饱和就会变得过时的静态基准，MORSE-500 是为了进化而构建的：其可控的生成管道支持创建任意具有挑战性的新实例，使其非常适合测试下一代模型的压力。初始实验证明，最新的顶级系统——包括各种Gemini 2.5 Pro和OpenAI o3以及强大的开源模型——在所有类别中均表现出显著的性能差距，尤其是在抽象和规划任务中的差距尤为显著。我们发布了完整的数据集、生成脚本和评估框架，以支持透明、可重复和前瞻性的多模态推理研究。 

---
# Coordinated Robustness Evaluation Framework for Vision-Language Models 

**Title (ZH)**: 视觉-语言模型协调稳健性评估框架 

**Authors**: Ashwin Ramesh Babu, Sajad Mousavi, Vineet Gundecha, Sahand Ghorbanpour, Avisek Naug, Antonio Guillen, Ricardo Luna Gutierrez, Soumyendu Sarkar  

**Link**: [PDF](https://arxiv.org/pdf/2506.05429)  

**Abstract**: Vision-language models, which integrate computer vision and natural language processing capabilities, have demonstrated significant advancements in tasks such as image captioning and visual question and answering. However, similar to traditional models, they are susceptible to small perturbations, posing a challenge to their robustness, particularly in deployment scenarios. Evaluating the robustness of these models requires perturbations in both the vision and language modalities to learn their inter-modal dependencies. In this work, we train a generic surrogate model that can take both image and text as input and generate joint representation which is further used to generate adversarial perturbations for both the text and image modalities. This coordinated attack strategy is evaluated on the visual question and answering and visual reasoning datasets using various state-of-the-art vision-language models. Our results indicate that the proposed strategy outperforms other multi-modal attacks and single-modality attacks from the recent literature. Our results demonstrate their effectiveness in compromising the robustness of several state-of-the-art pre-trained multi-modal models such as instruct-BLIP, ViLT and others. 

**Abstract (ZH)**: 视觉语言模型综合了计算机视觉和自然语言处理能力，在图像字幕和视觉问答等任务上取得了显著进展。然而，类似传统模型，它们对小规模扰动敏感，这给其鲁棒性带来了挑战，尤其是在部署场景中。评估这些模型的鲁棒性需要同时在视觉和语言模态上施加扰动，以学习其跨模态依赖关系。在此工作中，我们训练了一个通用的替代模型，该模型可以接受图像和文本作为输入，并生成联合表示，进一步用于为文本和图像模态生成对抗性扰动。这种协调攻击策略在视觉问答和视觉推理数据集上使用了多种最先进的视觉语言模型进行了评估。我们的结果显示，所提出的策略在应对多模态攻击和近期文献中的单模态攻击方面表现更优。我们的结果表明，该策略在多种最新的预训练多模态模型如instruct-BLIP、ViLT等上削弱了其鲁棒性。 

---
