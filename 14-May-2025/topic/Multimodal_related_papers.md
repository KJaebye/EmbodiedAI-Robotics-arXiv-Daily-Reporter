# A Comparative Study of Human Activity Recognition: Motion, Tactile, and multi-modal Approaches 

**Title (ZH)**: 人类活动识别的比较研究：运动、触觉及多模态方法 

**Authors**: Valerio Belcamino, Nhat Minh Dinh Le, Quan Khanh Luu, Alessandro Carfì, Van Anh Ho, Fulvio Mastrogiovanni  

**Link**: [PDF](https://arxiv.org/pdf/2505.08657)  

**Abstract**: Human activity recognition (HAR) is essential for effective Human-Robot Collaboration (HRC), enabling robots to interpret and respond to human actions. This study evaluates the ability of a vision-based tactile sensor to classify 15 activities, comparing its performance to an IMU-based data glove. Additionally, we propose a multi-modal framework combining tactile and motion data to leverage their complementary strengths. We examined three approaches: motion-based classification (MBC) using IMU data, tactile-based classification (TBC) with single or dual video streams, and multi-modal classification (MMC) integrating both. Offline validation on segmented datasets assessed each configuration's accuracy under controlled conditions, while online validation on continuous action sequences tested online performance. Results showed the multi-modal approach consistently outperformed single-modality methods, highlighting the potential of integrating tactile and motion sensing to enhance HAR systems for collaborative robotics. 

**Abstract (ZH)**: 基于视觉的触觉传感器在15项活动分类中的评估及其在人机协作中的多模态框架研究 

---
# CLTP: Contrastive Language-Tactile Pre-training for 3D Contact Geometry Understanding 

**Title (ZH)**: CLTP: 对比学习语言-触觉预训练以理解三维接触几何 

**Authors**: Wenxuan Ma, Xiaoge Cao, Yixiang Zhang, Chaofan Zhang, Shaobo Yang, Peng Hao, Bin Fang, Yinghao Cai, Shaowei Cui, Shuo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.08194)  

**Abstract**: Recent advancements in integrating tactile sensing with vision-language models (VLMs) have demonstrated remarkable potential for robotic multimodal perception. However, existing tactile descriptions remain limited to superficial attributes like texture, neglecting critical contact states essential for robotic manipulation. To bridge this gap, we propose CLTP, an intuitive and effective language tactile pretraining framework that aligns tactile 3D point clouds with natural language in various contact scenarios, thus enabling contact-state-aware tactile language understanding for contact-rich manipulation tasks. We first collect a novel dataset of 50k+ tactile 3D point cloud-language pairs, where descriptions explicitly capture multidimensional contact states (e.g., contact location, shape, and force) from the tactile sensor's perspective. CLTP leverages a pre-aligned and frozen vision-language feature space to bridge holistic textual and tactile modalities. Experiments validate its superiority in three downstream tasks: zero-shot 3D classification, contact state classification, and tactile 3D large language model (LLM) interaction. To the best of our knowledge, this is the first study to align tactile and language representations from the contact state perspective for manipulation tasks, providing great potential for tactile-language-action model learning. Code and datasets are open-sourced at this https URL. 

**Abstract (ZH)**: Recent advancements in 将触觉感知与视觉语言模型（VLMs）集成的进展在机器人多模态感知方面展现了巨大的潜力。然而，现有的触觉描述仍然局限于如纹理等表面属性，忽略了对于机器人操作至关重要的接触状态。为了解决这一问题，我们提出了CLTP，这是一种直观且有效的位置语言触觉预训练框架，该框架在各种接触场景中将触觉3D点云与自然语言对齐，从而实现感知丰富接触的接触状态感知触觉语言理解。我们首先收集了一个包含50K+触觉3D点云-语言对的新数据集，在这些描述中，从触觉传感器的角度明确捕捉到了多维度的接触状态（例如，接触位置、形状和力）。CLTP利用预对齐和冻结的视觉语言特征空间连接了整体的语言和触觉模态。实验在其后的三个下游任务中验证了其优越性：零样本3D分类、接触状态分类以及触觉3D大语言模型（LLM）交互。据我们所知，这是首次从接触状态视角将触觉和语言表示相结合进行操作任务的研究，为触觉-语言-动作模型的学习提供了巨大潜力。代码和数据集在该链接处开源。 

---
# Multimodal Assessment of Classroom Discourse Quality: A Text-Centered Attention-Based Multi-Task Learning Approach 

**Title (ZH)**: 基于文本中心注意力的多任务学习的课堂教学 discourse 质量的多模态评估方法 

**Authors**: Ruikun Hou, Babette Bühler, Tim Fütterer, Efe Bozkir, Peter Gerjets, Ulrich Trautwein, Enkelejda Kasneci  

**Link**: [PDF](https://arxiv.org/pdf/2505.07902)  

**Abstract**: Classroom discourse is an essential vehicle through which teaching and learning take place. Assessing different characteristics of discursive practices and linking them to student learning achievement enhances the understanding of teaching quality. Traditional assessments rely on manual coding of classroom observation protocols, which is time-consuming and costly. Despite many studies utilizing AI techniques to analyze classroom discourse at the utterance level, investigations into the evaluation of discursive practices throughout an entire lesson segment remain limited. To address this gap, our study proposes a novel text-centered multimodal fusion architecture to assess the quality of three discourse components grounded in the Global Teaching InSights (GTI) observation protocol: Nature of Discourse, Questioning, and Explanations. First, we employ attention mechanisms to capture inter- and intra-modal interactions from transcript, audio, and video streams. Second, a multi-task learning approach is adopted to jointly predict the quality scores of the three components. Third, we formulate the task as an ordinal classification problem to account for rating level order. The effectiveness of these designed elements is demonstrated through an ablation study on the GTI Germany dataset containing 92 videotaped math lessons. Our results highlight the dominant role of text modality in approaching this task. Integrating acoustic features enhances the model's consistency with human ratings, achieving an overall Quadratic Weighted Kappa score of 0.384, comparable to human inter-rater reliability (0.326). Our study lays the groundwork for the future development of automated discourse quality assessment to support teacher professional development through timely feedback on multidimensional discourse practices. 

**Abstract (ZH)**: 课堂话语是教学和学习进行的重要载体。评估话语实践的不同特征并将它们与学生学业成就联系起来，有助于提高对教学质量的理解。传统评估依赖于人工编译课堂观察协议，耗时且成本高。尽管许多研究利用AI技术在话语层面分析课堂话语，但关于整个教学片段中话语实践评估的研究仍然有限。为弥补这一空白，本研究提出了一种以文本为中心的多模态融合架构，评估Global Teaching InSights (GTI) 观察协议下的三种话语组件的质量：话语性质、提问和解释。首先，我们采用注意力机制捕捉转录、音频和视频流之间的跨模态和内模态交互。其次，采用多任务学习方法联合预测三种组件的质量分数。第三，我们将任务表述为序数分类问题，以考虑评分等级的顺序。通过在包含92节录数学课程的GTI德国数据集上的消融研究，证明了所设计元素的有效性。我们的结果突显了文本模态在处理此任务中的主导作用。结合声学特征提高了模型与人类评分的一致性，获得了总体的Quadratic Weighted Kappa评分为0.384，这一评分与人类评分者可靠性（0.326）相当。本研究为利用及时的多维度话语实践反馈支持教师专业发展奠定了基础。 

---
# Representation Learning with Mutual Influence of Modalities for Node Classification in Multi-Modal Heterogeneous Networks 

**Title (ZH)**: 多模态异构网络中模态间相互影响的表示学习在节点分类中的应用 

**Authors**: Jiafan Li, Jiaqi Zhu, Liang Chang, Yilin Li, Miaomiao Li, Yang Wang, Hongan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.07895)  

**Abstract**: Nowadays, numerous online platforms can be described as multi-modal heterogeneous networks (MMHNs), such as Douban's movie networks and Amazon's product review networks. Accurately categorizing nodes within these networks is crucial for analyzing the corresponding entities, which requires effective representation learning on nodes. However, existing multi-modal fusion methods often adopt either early fusion strategies which may lose the unique characteristics of individual modalities, or late fusion approaches overlooking the cross-modal guidance in GNN-based information propagation. In this paper, we propose a novel model for node classification in MMHNs, named Heterogeneous Graph Neural Network with Inter-Modal Attention (HGNN-IMA). It learns node representations by capturing the mutual influence of multiple modalities during the information propagation process, within the framework of heterogeneous graph transformer. Specifically, a nested inter-modal attention mechanism is integrated into the inter-node attention to achieve adaptive multi-modal fusion, and modality alignment is also taken into account to encourage the propagation among nodes with consistent similarities across all modalities. Moreover, an attention loss is augmented to mitigate the impact of missing modalities. Extensive experiments validate the superiority of the model in the node classification task, providing an innovative view to handle multi-modal data, especially when accompanied with network structures. 

**Abstract (ZH)**: 异构图神经网络中的跨模态注意力模型：Heterogeneous Graph Neural Network with Inter-Modal Attention (HGNN-IMA) 

---
# OMGM: Orchestrate Multiple Granularities and Modalities for Efficient Multimodal Retrieval 

**Title (ZH)**: OMGM： orchestrating 多种粒度和模态以实现高效的多模态检索 

**Authors**: Wei Yang, Jingjing Fu, Rui Wang, Jinyu Wang, Lei Song, Jiang Bian  

**Link**: [PDF](https://arxiv.org/pdf/2505.07879)  

**Abstract**: Vision-language retrieval-augmented generation (RAG) has become an effective approach for tackling Knowledge-Based Visual Question Answering (KB-VQA), which requires external knowledge beyond the visual content presented in images. The effectiveness of Vision-language RAG systems hinges on multimodal retrieval, which is inherently challenging due to the diverse modalities and knowledge granularities in both queries and knowledge bases. Existing methods have not fully tapped into the potential interplay between these elements. We propose a multimodal RAG system featuring a coarse-to-fine, multi-step retrieval that harmonizes multiple granularities and modalities to enhance efficacy. Our system begins with a broad initial search aligning knowledge granularity for cross-modal retrieval, followed by a multimodal fusion reranking to capture the nuanced multimodal information for top entity selection. A text reranker then filters out the most relevant fine-grained section for augmented generation. Extensive experiments on the InfoSeek and Encyclopedic-VQA benchmarks show our method achieves state-of-the-art retrieval performance and highly competitive answering results, underscoring its effectiveness in advancing KB-VQA systems. 

**Abstract (ZH)**: 视觉-语言检索增强生成（RAG）已成为处理知识导向的视觉问答（KB-VQA）的有效方法，KB-VQA需要超越图像呈现的视觉内容的外部知识。视觉-语言RAG系统的有效性取决于多模态检索，由于查询和知识库中的多种模态和知识粒度，这一过程本身具有挑战性。现有方法尚未充分挖掘这些元素之间的潜在互动。我们提出了一种多模态RAG系统，该系统采用从粗到细的多步检索方式，协调多个粒度和模态以提高效果。该系统首先进行广泛的初步检索以对齐知识粒度，为跨模态检索做准备，然后进行多模态融合排名重新排序，以捕捉关键实体的细微多模态信息。随后，文本重新排序器筛选出最相关的细粒度部分，以增强生成。在InfoSeek和Encyclopedic-VQA基准上的广泛实验表明，我们的方法在检索性能上达到最新水平，并且答案结果极具竞争力，凸显了其在推进KB-VQA系统方面的有效性。 

---
