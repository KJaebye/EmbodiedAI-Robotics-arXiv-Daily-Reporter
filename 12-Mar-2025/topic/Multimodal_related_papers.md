# Uni$\textbf{F}^2$ace: Fine-grained Face Understanding and Generation with Unified Multimodal Models 

**Title (ZH)**: Uni$\textbf{F}^2$ace: 统一多模态模型下的细粒度面部理解和生成 

**Authors**: Junzhe Li, Xuerui Qiu, Linrui Xu, Liya Guo, Delin Qu, Tingting Long, Chun Fan, Ming Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.08120)  

**Abstract**: Unified multimodal models (UMMs) have emerged as a powerful paradigm in foundational computer vision research, demonstrating significant potential in both image understanding and generation. However, existing research in the face domain primarily focuses on $\textbf{coarse}$ facial attribute understanding, with limited capacity to handle $\textbf{fine-grained}$ facial attributes and without addressing generation capabilities. To overcome these limitations, we propose Uni$\textbf{F}^2$ace, the first UMM tailored specifically for fine-grained face understanding and generation. In general, we train Uni$\textbf{F}^2$ace on a self-constructed, specialized dataset utilizing two mutually beneficial diffusion techniques and a two-level mixture-of-experts architecture. Specifically, we first build a large-scale facial dataset, Uni$\textbf{F}^2$ace-130K, which contains 130K image-text pairs with one million question-answering pairs that span a wide range of facial attributes. Second, we establish a theoretical connection between discrete diffusion score matching and masked generative models, optimizing both evidence lower bounds simultaneously, which significantly improves the model's ability to synthesize facial details. Finally, we introduce both token-level and sequence-level mixture-of-experts, enabling efficient fine-grained representation learning for both understanding and generation tasks. Extensive experiments on Uni$\textbf{F}^2$ace-130K demonstrate that Uni$\textbf{F}^2$ace outperforms existing UMMs and generative models, achieving superior performance across both understanding and generation tasks. 

**Abstract (ZH)**: 统一多模态模型（UMMs）在基础计算机视觉研究中 emerged as a powerful paradigm, demonstrating significant potential in both image understanding and generation. However, existing research in the face domain primarily focuses on coarse facial attribute understanding, with limited capacity to handle fine-grained facial attributes and without addressing generation capabilities. To overcome these limitations, we propose UniF²ace, the first UMM tailored specifically for fine-grained face understanding and generation. In general, we train UniF²ace on a self-constructed, specialized dataset utilizing two mutually beneficial diffusion techniques and a two-level mixture-of-experts architecture. Specifically, we first build a large-scale facial dataset, UniF²ace-130K, which contains 130K image-text pairs with one million question-answering pairs that span a wide range of facial attributes. Second, we establish a theoretical connection between discrete diffusion score matching and masked generative models, optimizing both evidence lower bounds simultaneously, which significantly improves the model's ability to synthesize facial details. Finally, we introduce both token-level and sequence-level mixture-of-experts, enabling efficient fine-grained representation learning for both understanding and generation tasks. Extensive experiments on UniF²ace-130K demonstrate that UniF²ace outperforms existing UMMs and generative models, achieving superior performance across both understanding and generation tasks. 

---
# Continual Learning for Multiple Modalities 

**Title (ZH)**: 多模态持续学习 

**Authors**: Hyundong Jin, Eunwoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2503.08064)  

**Abstract**: Continual learning aims to learn knowledge of tasks observed in sequential time steps while mitigating the forgetting of previously learned knowledge. Existing methods were proposed under the assumption of learning a single modality (e.g., image) over time, which limits their applicability in scenarios involving multiple modalities. In this work, we propose a novel continual learning framework that accommodates multiple modalities (image, video, audio, depth, and text). We train a model to align various modalities with text, leveraging its rich semantic information. However, this increases the risk of forgetting previously learned knowledge, exacerbated by the differing input traits of each task. To alleviate the overwriting of the previous knowledge of modalities, we propose a method for aggregating knowledge within and across modalities. The aggregated knowledge is obtained by assimilating new information through self-regularization within each modality and associating knowledge between modalities by prioritizing contributions from relevant modalities. Furthermore, we propose a strategy that re-aligns the embeddings of modalities to resolve biased alignment between modalities. We evaluate the proposed method in a wide range of continual learning scenarios using multiple datasets with different modalities. Extensive experiments demonstrate that ours outperforms existing methods in the scenarios, regardless of whether the identity of the modality is given. 

**Abstract (ZH)**: 多模态持续学习框架：融合图像、视频、音频、深度和文本信息以减轻遗忘风险 

---
# Crowdsource, Crawl, or Generate? Creating SEA-VL, a Multicultural Vision-Language Dataset for Southeast Asia 

**Title (ZH)**: 众包、爬取或生成？创造SEA-VL，一个面向东南亚的多文化视觉语言数据集 

**Authors**: Samuel Cahyawijaya, Holy Lovenia, Joel Ruben Antony Moniz, Tack Hwa Wong, Mohammad Rifqi Farhansyah, Thant Thiri Maung, Frederikus Hudi, David Anugraha, Muhammad Ravi Shulthan Habibi, Muhammad Reza Qorib, Amit Agarwal, Joseph Marvin Imperial, Hitesh Laxmichand Patel, Vicky Feliren, Bahrul Ilmi Nasution, Manuel Antonio Rufino, Genta Indra Winata, Rian Adam Rajagede, Carlos Rafael Catalan, Mohamed Fazli Imam, Priyaranjan Pattnayak, Salsabila Zahirah Pranida, Kevin Pratama, Yeshil Bangera, Adisai Na-Thalang, Patricia Nicole Monderin, Yueqi Song, Christian Simon, Lynnette Hui Xian Ng, Richardy Lobo' Sapan, Taki Hasan Rafi, Bin Wang, Supryadi, Kanyakorn Veerakanjana, Piyalitt Ittichaiwong, Matthew Theodore Roque, Karissa Vincentio, Takdanai Kreangphet, Phakphum Artkaew, Kadek Hendrawan Palgunadi, Yanzhi Yu, Rochana Prih Hastuti, William Nixon, Mithil Bangera, Adrian Xuan Wei Lim, Aye Hninn Khine, Hanif Muhammad Zhafran, Teddy Ferdinan, Audra Aurora Izzani, Ayushman Singh, Evan, Jauza Akbar Krito, Michael Anugraha, Fenal Ashokbhai Ilasariya, Haochen Li, John Amadeo Daniswara, Filbert Aurelian Tjiaranata, Eryawan Presma Yulianrifat, Can Udomcharoenchaikit, Fadil Risdian Ansori, Mahardika Krisna Ihsani, Giang Nguyen, Anab Maulana Barik, Dan John Velasco, Rifo Ahmad Genadi, Saptarshi Saha, Chengwei Wei, Isaiah Flores, Kenneth Ko Han Chen, Anjela Gail Santos, Wan Shen Lim, Kaung Si Phyo, Tim Santos, Meisyarah Dwiastuti, Jiayun Luo, Jan Christian Blaise Cruz, Ming Shan Hee, Ikhlasul Akmal Hanif, M.Alif Al Hakim, Muhammad Rizky Sya'ban, Kun Kerdthaisong, Lester James V. Miranda, Fajri Koto, Tirana Noor Fatyanosa, Alham Fikri Aji, Jostin Jerico Rosal, Jun Kevin, Robert Wijaya, Onno P. Kampman, Ruochen Zhang, Börje F. Karlsson, Peerat Limkonchotiwat  

**Link**: [PDF](https://arxiv.org/pdf/2503.07920)  

**Abstract**: Southeast Asia (SEA) is a region of extraordinary linguistic and cultural diversity, yet it remains significantly underrepresented in vision-language (VL) research. This often results in artificial intelligence (AI) models that fail to capture SEA cultural nuances. To fill this gap, we present SEA-VL, an open-source initiative dedicated to developing high-quality, culturally relevant data for SEA languages. By involving contributors from SEA countries, SEA-VL aims to ensure better cultural relevance and diversity, fostering greater inclusivity of underrepresented languages in VL research. Beyond crowdsourcing, our initiative goes one step further in the exploration of the automatic collection of culturally relevant images through crawling and image generation. First, we find that image crawling achieves approximately ~85% cultural relevance while being more cost- and time-efficient than crowdsourcing. Second, despite the substantial progress in generative vision models, synthetic images remain unreliable in accurately reflecting SEA cultures. The generated images often fail to reflect the nuanced traditions and cultural contexts of the region. Collectively, we gather 1.28M SEA culturally-relevant images, more than 50 times larger than other existing datasets. Through SEA-VL, we aim to bridge the representation gap in SEA, fostering the development of more inclusive AI systems that authentically represent diverse cultures across SEA. 

**Abstract (ZH)**: 东南亚（SEA）地区的语言和文化多样性突出，但在视觉语言（VL）研究中却显著欠代表。这常常导致人工智能（AI）模型无法捕捉到SEA的文化细微差别。为填补这一空白，我们提出SEA-VL这一开源项目，旨在为SEA语言开发高质量、文化相关的数据。通过来自SEA国家的贡献者，SEA-VL旨在确保更好的文化相关性和多样性，促进在VL研究中更广泛地包容欠代表的语言。除了众包，我们的倡议在通过爬取和图像生成自动收集文化相关图像方面更进一步。首先，我们发现图像爬取实现了约85%的文化相关性，且在成本和时间效率上优于众包。其次，尽管生成式视觉模型取得了重大进展，合成图像依然难以准确反映SEA文化。生成的图像往往无法体现该地区的细腻传统和文化背景。通过SEA-VL，我们共收集了1,280,000张SEA文化相关图像，规模超过现有其他数据集的50倍。通过SEA-VL，我们旨在填补SEA的代表性缺口，推动开发更包容的AI系统，真实地代表东南亚多元文化。 

---
# Video Action Differencing 

**Title (ZH)**: 视频动作差异分析 

**Authors**: James Burgess, Xiaohan Wang, Yuhui Zhang, Anita Rau, Alejandro Lozano, Lisa Dunlap, Trevor Darrell, Serena Yeung-Levy  

**Link**: [PDF](https://arxiv.org/pdf/2503.07860)  

**Abstract**: How do two individuals differ when performing the same action? In this work, we introduce Video Action Differencing (VidDiff), the novel task of identifying subtle differences between videos of the same action, which has many applications, such as coaching and skill learning. To enable development on this new task, we first create VidDiffBench, a benchmark dataset containing 549 video pairs, with human annotations of 4,469 fine-grained action differences and 2,075 localization timestamps indicating where these differences occur. Our experiments demonstrate that VidDiffBench poses a significant challenge for state-of-the-art large multimodal models (LMMs), such as GPT-4o and Qwen2-VL. By analyzing failure cases of LMMs on VidDiffBench, we highlight two key challenges for this task: localizing relevant sub-actions over two videos and fine-grained frame comparison. To overcome these, we propose the VidDiff method, an agentic workflow that breaks the task into three stages: action difference proposal, keyframe localization, and frame differencing, each stage utilizing specialized foundation models. To encourage future research in this new task, we release the benchmark at this https URL and code at this http URL. 

**Abstract (ZH)**: 两人在执行相同动作时有何不同？在本工作中，我们介绍了Video Action Differencing (VidDiff)这一新颖任务，旨在识别相同动作视频之间的细微差异，该任务在教练和技能学习等领域有广泛的应用。为了推进这一新任务的发展，我们首先创建了VidDiffBench基准数据集，包含549对视频，附有人类标注的4,469个细粒度动作差异和2,075个定位时间戳，指示这些差异发生的位置。实验结果表明，VidDiffBench对当前最先进的大型多模态模型（LMMs）如GPT-4o和Qwen2-VL构成了重大挑战。通过对LMMs在VidDiffBench上的失败案例的分析，我们指出了这一任务中的两个关键挑战：在两段视频中定位相关子动作和细粒度帧比较。为了克服这些挑战，我们提出了VidDiff方法，这是一种具备代理性的工作流，将任务分解为三个阶段：动作差异提议、关键帧定位和帧差异比较，每个阶段都利用专门的基础模型。为了促进未来对该新任务的研究，我们在https://链接和http://链接处发布了基准数据集和代码。 

---
# Data Foundations for Large Scale Multimodal Clinical Foundation Models 

**Title (ZH)**: 大型多模态临床基础模型的数据基础 

**Authors**: Wei Dai, Peilin Chen, Malinda Lu, Daniel Li, Haowen Wei, Hejie Cui, Paul Pu Liang  

**Link**: [PDF](https://arxiv.org/pdf/2503.07667)  

**Abstract**: Recent advances in clinical AI have enabled remarkable progress across many clinical domains. However, existing benchmarks and models are primarily limited to a small set of modalities and tasks, which hinders the development of large-scale multimodal methods that can make holistic assessments of patient health and well-being. To bridge this gap, we introduce Clinical Large-Scale Integrative Multimodal Benchmark (CLIMB), a comprehensive clinical benchmark unifying diverse clinical data across imaging, language, temporal, and graph modalities. CLIMB comprises 4.51 million patient samples totaling 19.01 terabytes distributed across 2D imaging, 3D video, time series, graphs, and multimodal data. Through extensive empirical evaluation, we demonstrate that multitask pretraining significantly improves performance on understudied domains, achieving up to 29% improvement in ultrasound and 23% in ECG analysis over single-task learning. Pretraining on CLIMB also effectively improves models' generalization capability to new tasks, and strong unimodal encoder performance translates well to multimodal performance when paired with task-appropriate fusion strategies. Our findings provide a foundation for new architecture designs and pretraining strategies to advance clinical AI research. Code is released at this https URL. 

**Abstract (ZH)**: 近期临床AI的发展在许多临床领域取得了显著进展。然而，现有的基准和模型主要局限于少量的模态和任务，这阻碍了能够全面评估患者健康和福祉的大规模多模态方法的发展。为了弥合这一差距，我们介绍了Clinical Large-Scale Integrative Multimodal Benchmark (CLIMB)，这是一个综合性的临床基准，统一了来自影像、语言、时间序列和图等多种模态的多样临床数据。CLIMB 包含总计 4.51 百万患者的样本，数据量达到 19.01 太字节，数据分布在 2D 影像、3D 视频、时间序列、图形和多模态数据中。通过广泛的实验评估，我们证明了多任务预训练显著提高了对未研究领域的性能，分别在超声和心电信号分析中提高了多达 29% 和 23%。在 CLIMB 上进行预训练还有效提高了模型对新任务的泛化能力，而且在适当融合策略的配合下，单一模态编码器的强性能也能很好地转换为多模态性能。我们的研究结果为新的架构设计和预训练策略提供了基础，以推进临床AI研究。代码发布在该网址：<https://>。 

---
# An Optimization Algorithm for Multimodal Data Alignment 

**Title (ZH)**: 多模态数据对齐的优化算法 

**Authors**: Wei Zhang, Xinyue Wang, Lan Yu, Shi Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.07636)  

**Abstract**: In the data era, the integration of multiple data types, known as multimodality, has become a key area of interest in the research community. This interest is driven by the goal to develop cutting edge multimodal models capable of serving as adaptable reasoning engines across a wide range of modalities and domains. Despite the fervent development efforts, the challenge of optimally representing different forms of data within a single unified latent space a crucial step for enabling effective multimodal reasoning has not been fully addressed. To bridge this gap, we introduce AlignXpert, an optimization algorithm inspired by Kernel CCA crafted to maximize the similarities between N modalities while imposing some other constraints. This work demonstrates the impact on improving data representation for a variety of reasoning tasks, such as retrieval and classification, underlining the pivotal importance of data representation. 

**Abstract (ZH)**: 数据时代，多模态数据类型的整合已成为研究社区的一个关键研究领域。通过对不同形式的数据在单一统一潜在空间中的最优表示进行优化，以促进有效的多模态推理，尽管进行了广泛的努力，这一挑战尚未得到充分解决。为了弥合这一差距，我们引入了AlignXpert，这是一种受核CCA启发的优化算法，旨在最大化N种模态之间的相似性并施加其他约束。本研究展示了提高数据表示以进行多种推理任务（如检索和分类）的效果，突显了数据表示的重要性。 

---
