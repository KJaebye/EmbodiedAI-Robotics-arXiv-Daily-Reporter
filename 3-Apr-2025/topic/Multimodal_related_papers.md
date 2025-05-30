# TransforMerger: Transformer-based Voice-Gesture Fusion for Robust Human-Robot Communication 

**Title (ZH)**: TransforMerger: 基于Transformer的语音-手势融合技术以增强人机通信 

**Authors**: Petr Vanc, Karla Stepanova  

**Link**: [PDF](https://arxiv.org/pdf/2504.01708)  

**Abstract**: As human-robot collaboration advances, natural and flexible communication methods are essential for effective robot control. Traditional methods relying on a single modality or rigid rules struggle with noisy or misaligned data as well as with object descriptions that do not perfectly fit the predefined object names (e.g. 'Pick that red object'). We introduce TransforMerger, a transformer-based reasoning model that infers a structured action command for robotic manipulation based on fused voice and gesture inputs. Our approach merges multimodal data into a single unified sentence, which is then processed by the language model. We employ probabilistic embeddings to handle uncertainty and we integrate contextual scene understanding to resolve ambiguous references (e.g., gestures pointing to multiple objects or vague verbal cues like "this"). We evaluate TransforMerger in simulated and real-world experiments, demonstrating its robustness to noise, misalignment, and missing information. Our results show that TransforMerger outperforms deterministic baselines, especially in scenarios requiring more contextual knowledge, enabling more robust and flexible human-robot communication. Code and datasets are available at: this http URL. 

**Abstract (ZH)**: 随着人机协作的发展，自然灵活的通信方法对于有效的机器人控制至关重要。传统的单一模态或刚性规则的方法难以处理噪声或错位的数据，以及不符合预定义对象名称的对象描述（例如，“捡起那个红色的对象”）。我们提出了基于变换器的TransforMerger模型，该模型基于融合的语音和手势输入推断出结构化的动作指令。我们的方法将多模态数据合并为一个统一的句子，然后由语言模型处理。我们采用概率嵌入来处理不确定性，并结合上下文场景理解来解决模棱两可的引用（例如，手势指向多个对象或模糊的口头提示“这个”）。我们在模拟和真实世界实验中评估了TransforMerger，证明了其对噪声、错位和缺失信息的鲁棒性。实验结果表明，TransforMerger在需要更多上下文知识的场景中优于确定性基线，从而实现了更 robust 和灵活的人机通信。代码和数据集可在以下链接获取：this http URL。 

---
# Leveraging Embedding Techniques in Multimodal Machine Learning for Mental Illness Assessment 

**Title (ZH)**: 利用嵌入技术在多模态机器学习中的精神疾病评估 

**Authors**: Abdelrahaman A. Hassan, Abdelrahman A. Ali, Aya E. Fouda, Radwa J. Hanafy, Mohammed E. Fouda  

**Link**: [PDF](https://arxiv.org/pdf/2504.01767)  

**Abstract**: The increasing global prevalence of mental disorders, such as depression and PTSD, requires objective and scalable diagnostic tools. Traditional clinical assessments often face limitations in accessibility, objectivity, and consistency. This paper investigates the potential of multimodal machine learning to address these challenges, leveraging the complementary information available in text, audio, and video data. Our approach involves a comprehensive analysis of various data preprocessing techniques, including novel chunking and utterance-based formatting strategies. We systematically evaluate a range of state-of-the-art embedding models for each modality and employ Convolutional Neural Networks (CNNs) and Bidirectional LSTM Networks (BiLSTMs) for feature extraction. We explore data-level, feature-level, and decision-level fusion techniques, including a novel integration of Large Language Model (LLM) predictions. We also investigate the impact of replacing Multilayer Perceptron classifiers with Support Vector Machines. We extend our analysis to severity prediction using PHQ-8 and PCL-C scores and multi-class classification (considering co-occurring conditions). Our results demonstrate that utterance-based chunking significantly improves performance, particularly for text and audio modalities. Decision-level fusion, incorporating LLM predictions, achieves the highest accuracy, with a balanced accuracy of 94.8% for depression and 96.2% for PTSD detection. The combination of CNN-BiLSTM architectures with utterance-level chunking, coupled with the integration of external LLM, provides a powerful and nuanced approach to the detection and assessment of mental health conditions. Our findings highlight the potential of MMML for developing more accurate, accessible, and personalized mental healthcare tools. 

**Abstract (ZH)**: 全球精神障碍（如抑郁和PTSD）的患病率不断增加，亟需客观且可扩展的诊断工具。传统临床评估往往面临可及性、客观性和一致性方面的限制。本文探讨了多模态机器学习的潜在应用，以应对这些挑战，利用文本、音频和视频数据中的互补信息。我们的方法包括对各种数据预处理技术的全面分析，包括新颖的片段化和基于陈述的格式化策略。我们系统性地评估了每种模态的多项前沿嵌入模型，并使用卷积神经网络（CNNs）和双向长短期记忆网络（BiLSTMs）进行特征提取。我们探索了数据级、特征级和决策级融合技术，包括新型大型语言模型（LLM）预测的集成。我们还研究了用支持向量机（SVM）替换多层感知机（MLP）分类器的影响。我们还将分析扩展到使用PHQ-8和PCL-C评分进行严重程度预测和多类分类（考虑共病情况）。结果显示，基于陈述的片段化显著提高了性能，特别是对于文本和音频模态。决策级融合结合LLM预测达到了最高的准确性，抑郁症检测的平衡准确率为94.8%，PTSD检测的平衡准确率为96.2%。结合基于陈述的片段化与CNN-BiLSTM架构，并集成外部LLM，提供了一种强大且细腻的精神健康状况检测和评估方法。我们的研究结果突显了多模态机器学习（MMML）在开发更准确、更可访问和更具个性化的心理健康护理工具方面的潜力。 

---
# Text Speaks Louder than Vision: ASCII Art Reveals Textual Biases in Vision-Language Models 

**Title (ZH)**: 文字胜过图像：ASCII艺术揭示了视觉语言模型中的文本偏见 

**Authors**: Zhaochen Wang, Yujun Cai, Zi Huang, Bryan Hooi, Yiwei Wang, Ming-Hsuan Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.01589)  

**Abstract**: Vision-language models (VLMs) have advanced rapidly in processing multimodal information, but their ability to reconcile conflicting signals across modalities remains underexplored. This work investigates how VLMs process ASCII art, a unique medium where textual elements collectively form visual patterns, potentially creating semantic-visual conflicts. We introduce a novel evaluation framework that systematically challenges five state-of-the-art models (including GPT-4o, Claude, and Gemini) using adversarial ASCII art, where character-level semantics deliberately contradict global visual patterns. Our experiments reveal a strong text-priority bias: VLMs consistently prioritize textual information over visual patterns, with visual recognition ability declining dramatically as semantic complexity increases. Various mitigation attempts through visual parameter tuning and prompt engineering yielded only modest improvements, suggesting that this limitation requires architectural-level solutions. These findings uncover fundamental flaws in how current VLMs integrate multimodal information, providing important guidance for future model development while highlighting significant implications for content moderation systems vulnerable to adversarial examples. 

**Abstract (ZH)**: 视觉-语言模型（VLMs）在处理多模态信息方面取得了快速进展，但在调和跨模态的冲突信号方面仍存在不足。本研究探讨了VLMs处理ASCII艺术的方法，ASCII艺术作为一种独特的媒介，其中文本文本元素共同形成视觉图案，可能会产生语义-视觉冲突。我们提出了一个新的评估框架，系统地使用对抗性ASCII艺术（其中字符级语义故意与全局视觉模式相矛盾）来挑战五种最先进的模型（包括GPT-4o、Claude和Gemini）。实验结果显示了强烈的文本优先偏见：VLMs始终优先处理文本信息而非视觉模式，在语义复杂性增加时，视觉识别能力急剧下降。尽管通过视觉参数调整和提示工程进行了多种缓解尝试，但仅取得了微小改进，这表明这一限制需要在架构层面寻找解决方案。这些发现揭示了当前VLMs整合多模态信息的基本缺陷，为未来的模型开发提供了重要指导，并强调了在对抗性示例面前内容审核系统的重要影响。 

---
# COST: Contrastive One-Stage Transformer for Vision-Language Small Object Tracking 

**Title (ZH)**: COST: 对比式一步Transformer视觉-语言小目标跟踪 

**Authors**: Chunhui Zhang, Li Liu, Jialin Gao, Xin Sun, Hao Wen, Xi Zhou, Shiming Ge, Yanfeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.01321)  

**Abstract**: Transformer has recently demonstrated great potential in improving vision-language (VL) tracking algorithms. However, most of the existing VL trackers rely on carefully designed mechanisms to perform the multi-stage multi-modal fusion. Additionally, direct multi-modal fusion without alignment ignores distribution discrepancy between modalities in feature space, potentially leading to suboptimal representations. In this work, we propose COST, a contrastive one-stage transformer fusion framework for VL tracking, aiming to learn semantically consistent and unified VL representations. Specifically, we introduce a contrastive alignment strategy that maximizes mutual information (MI) between a video and its corresponding language description. This enables effective cross-modal alignment, yielding semantically consistent features in the representation space. By leveraging a visual-linguistic transformer, we establish an efficient multi-modal fusion and reasoning mechanism, empirically demonstrating that a simple stack of transformer encoders effectively enables unified VL representations. Moreover, we contribute a newly collected VL tracking benchmark dataset for small object tracking, named VL-SOT500, with bounding boxes and language descriptions. Our dataset comprises two challenging subsets, VL-SOT230 and VL-SOT270, dedicated to evaluating generic and high-speed small object tracking, respectively. Small object tracking is notoriously challenging due to weak appearance and limited features, and this dataset is, to the best of our knowledge, the first to explore the usage of language cues to enhance visual representation for small object tracking. Extensive experiments demonstrate that COST achieves state-of-the-art performance on five existing VL tracking datasets, as well as on our proposed VL-SOT500 dataset. Source codes and dataset will be made publicly available. 

**Abstract (ZH)**: 基于变换器的一阶段视图语言跟踪对比融合框架(COST) 

---
