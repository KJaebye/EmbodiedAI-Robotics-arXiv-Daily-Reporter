# OmniNova:A General Multimodal Agent Framework 

**Title (ZH)**: OmniNova：一个通用多模态代理框架 

**Authors**: Pengfei Du  

**Link**: [PDF](https://arxiv.org/pdf/2503.20028)  

**Abstract**: The integration of Large Language Models (LLMs) with specialized tools presents new opportunities for intelligent automation systems. However, orchestrating multiple LLM-driven agents to tackle complex tasks remains challenging due to coordination difficulties, inefficient resource utilization, and inconsistent information flow. We present OmniNova, a modular multi-agent automation framework that combines language models with specialized tools such as web search, crawling, and code execution capabilities. OmniNova introduces three key innovations: (1) a hierarchical multi-agent architecture with distinct coordinator, planner, supervisor, and specialist agents; (2) a dynamic task routing mechanism that optimizes agent deployment based on task complexity; and (3) a multi-layered LLM integration system that allocates appropriate models to different cognitive requirements. Our evaluations across 50 complex tasks in research, data analysis, and web interaction domains demonstrate that OmniNova outperforms existing frameworks in task completion rate (87\% vs. baseline 62\%), efficiency (41\% reduced token usage), and result quality (human evaluation score of 4.2/5 vs. baseline 3.1/5). We contribute both a theoretical framework for multi-agent system design and an open-source implementation that advances the state-of-the-art in LLM-based automation systems. 

**Abstract (ZH)**: 大型语言模型与专用工具的集成为智能自动化系统带来了新的机遇。然而，由于协调难题、资源利用不充分以及信息流动不一致，多代理系统处理复杂任务仍然具有挑战性。我们提出了OmniNova，这是一种模块化的多代理自动化框架，将语言模型与网络搜索、爬取和代码执行等专用工具相结合。OmniNova 引入了三项关键创新：(1) 具有不同协调员、计划者、监督者和专家代理的层次化多代理架构；(2) 动态任务路由机制，该机制根据任务复杂性优化代理部署；(3) 多层大型语言模型集成系统，将合适的模型分配到不同的认知需求。我们在研究、数据分析和网络交互领域的50项复杂任务评估中表明，OmniNova 在任务完成率（87% vs. 对照组 62%）、效率（减少41%的令牌使用量）和结果质量（人类评估得分为4.2/5 vs. 对照组 3.1/5）方面均优于现有框架。我们不仅贡献了一个多代理系统设计的理论框架，还提供了一个开源实现，该实现推动了基于大型语言模型的自动化系统的发展。 

---
# LEGO-Puzzles: How Good Are MLLMs at Multi-Step Spatial Reasoning? 

**Title (ZH)**: LEGO-拼图：MLLMs在多步空间推理方面表现如何？ 

**Authors**: Kexian Tang, Junyao Gao, Yanhong Zeng, Haodong Duan, Yanan Sun, Zhening Xing, Wenran Liu, Kaifeng Lyu, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.19990)  

**Abstract**: Multi-step spatial reasoning entails understanding and reasoning about spatial relationships across multiple sequential steps, which is crucial for tackling complex real-world applications, such as robotic manipulation, autonomous navigation, and automated assembly. To assess how well current Multimodal Large Language Models (MLLMs) have acquired this fundamental capability, we introduce \textbf{LEGO-Puzzles}, a scalable benchmark designed to evaluate both \textbf{spatial understanding} and \textbf{sequential reasoning} in MLLMs through LEGO-based tasks. LEGO-Puzzles consists of 1,100 carefully curated visual question-answering (VQA) samples spanning 11 distinct tasks, ranging from basic spatial understanding to complex multi-step reasoning. Based on LEGO-Puzzles, we conduct a comprehensive evaluation of state-of-the-art MLLMs and uncover significant limitations in their spatial reasoning capabilities: even the most powerful MLLMs can answer only about half of the test cases, whereas human participants achieve over 90\% accuracy. In addition to VQA tasks, we evaluate MLLMs' abilities to generate LEGO images following assembly illustrations. Our experiments show that only Gemini-2.0-Flash and GPT-4o exhibit a limited ability to follow these instructions, while other MLLMs either replicate the input image or generate completely irrelevant outputs. Overall, LEGO-Puzzles exposes critical deficiencies in existing MLLMs' spatial understanding and sequential reasoning capabilities, and underscores the need for further advancements in multimodal spatial reasoning. 

**Abstract (ZH)**: 多步空间推理涉及理解并推理多个连续步骤的空间关系，这对于解决复杂的现实世界应用，如机器人操作、自主导航和自动化装配至关重要。为了评估当前多模态大型语言模型（MLLMs）在这一基本能力上的掌握情况，我们引入了基于LEGO的可扩展基准LEGO-Puzzles，用于评估MLLMs的空间理解和序列推理能力。LEGO-Puzzles包含1100个精心挑选的LEGO视觉问答（VQA）样本，涵盖从基本的空间理解到复杂的多步推理的11种不同任务。基于LEGO-Puzzles，我们全面评估了最先进的MLLMs，并揭示了它们在空间推理能力方面的显著局限：即使是最强大的MLLMs也只能正确回答大约一半的测试案例，而人类参与者则能实现超过90%的准确率。除了VQA任务外，我们还评估了MLLMs生成遵循装配示意图的LEGO图片的能力。实验结果显示，只有Gemini-2.0-Flash和GPT-4o表现出有限的跟随这些指令的能力，而其他MLLMs要么复制输入图片，要么生成完全不相关的输出。总体而言，LEGO-Puzzles暴露了现有MLLMs在空间理解和序列推理能力上的关键缺陷，并强调了在多模态空间推理方面进一步发展的必要性。 

---
# Mobile-MMLU: A Mobile Intelligence Language Understanding Benchmark 

**Title (ZH)**: 移动智能语言理解基准-Mobile-MMLU 

**Authors**: Sondos Mahmoud Bsharat, Mukul Ranjan, Aidar Myrzakhan, Jiacheng Liu, Bowei Guo, Shengkun Tang, Zhuang Liu, Yuanzhi Li, Zhiqiang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2503.20786)  

**Abstract**: Rapid advancements in large language models (LLMs) have increased interest in deploying them on mobile devices for on-device AI applications. Mobile users interact differently with LLMs compared to desktop users, creating unique expectations and data biases. Current benchmark datasets primarily target at server and desktop environments, and there is a notable lack of extensive datasets specifically designed for mobile contexts. Additionally, mobile devices face strict limitations in storage and computing resources, constraining model size and capabilities, thus requiring optimized efficiency and prioritized knowledge. To address these challenges, we introduce Mobile-MMLU, a large-scale benchmark dataset tailored for mobile intelligence. It consists of 16,186 questions across 80 mobile-related fields, designed to evaluate LLM performance in realistic mobile scenarios. A challenging subset, Mobile-MMLU-Pro, provides advanced evaluation similar in size to MMLU-Pro but significantly more difficult than our standard full set. Both benchmarks use multiple-choice, order-invariant questions focused on practical mobile interactions, such as recipe suggestions, travel planning, and essential daily tasks. The dataset emphasizes critical mobile-specific metrics like inference latency, energy consumption, memory usage, and response quality, offering comprehensive insights into model performance under mobile constraints. Moreover, it prioritizes privacy and adaptability, assessing models' ability to perform on-device processing, maintain user privacy, and adapt to personalized usage patterns. Mobile-MMLU family offers a standardized framework for developing and comparing mobile-optimized LLMs, enabling advancements in productivity and decision-making within mobile computing environments. Our code and data are available at: this https URL. 

**Abstract (ZH)**: 快速发展的大规模语言模型（LLMs）增加了在移动设备上部署它们以进行本地AI应用的兴趣。移动用户与LLMs的交互方式不同于桌面用户，创造了独特的需求和数据偏差。当前的标准基准数据集主要针对服务器和桌面环境，而专门针对移动场景的广泛数据集则明显不足。此外，移动设备在存储和计算资源方面受到严格限制，限制了模型的规模和能力，因此需要优化效率和优先考虑知识。为了解决这些挑战，我们引入了Mobile-MMLU，这是一种定制于移动智能的大规模基准数据集。它包含了16,186个问题，覆盖了80个移动相关领域，旨在评估LLM在现实移动场景中的性能。Mobile-MMLU-Pro是一个具有挑战性的子集，提供了类似MMLU-Pro的复杂评估，但难度显著高于我们标准的完整集。基准数据集使用多项选择、顺序不变的问题，集中在实用的移动交互上，如食谱建议、旅行规划和日常任务。该数据集强调关键的移动特定指标，如推理延迟、能耗、内存使用和响应质量，提供了在移动约束条件下模型性能的全面洞察。此外，它优先考虑隐私和适应性，评估模型在设备上处理、维护用户隐私和适应个性化使用模式的能力。Mobile-MMLU家族提供了一个标准化框架，用于开发和比较移动优化的LLMs，在移动计算环境中促进生产率和决策的提升。我们的代码和数据可在以下网址获取：this https URL。 

---
