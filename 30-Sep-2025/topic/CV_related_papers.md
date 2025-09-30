# Path Diffuser: Diffusion Model for Data-Driven Traffic Simulator 

**Title (ZH)**: 数据驱动交通模拟器中的路径扩散模型 

**Authors**: Da Saem Lee, Akash Karthikeyan, Yash Vardhan Pant, Sebastian Fischmeister  

**Link**: [PDF](https://arxiv.org/pdf/2509.24995)  

**Abstract**: Simulating diverse and realistic traffic scenarios is critical for developing and testing autonomous planning. Traditional rule-based planners lack diversity and realism, while learning-based simulators often replay, forecast, or edit scenarios using historical agent trajectories. However, they struggle to generate new scenarios, limiting scalability and diversity due to their reliance on fully annotated logs and historical data. Thus, a key challenge for a learning-based simulator's performance is that it requires agents' past trajectories and pose information in addition to map data, which might not be available for all agents on the this http URL which, generated scenarios often produce unrealistic trajectories that deviate from drivable areas, particularly under out-of-distribution (OOD) map scenes (e.g., curved roads). To address this, we propose Path Diffuser (PD): a two-stage, diffusion model for generating agent pose initializations and their corresponding trajectories conditioned on the map, free of any historical context of agents' trajectories. Furthermore, PD incorporates a motion primitive-based prior, leveraging Frenet frame candidate trajectories to enhance diversity while ensuring road-compliant trajectory generation. We also explore various design choices for modeling complex multi-agent interactions. We demonstrate the effectiveness of our method through extensive experiments on the Argoverse2 Dataset and additionally evaluate the generalizability of the approach on OOD map variants. Notably, Path Diffuser outperforms the baseline methods by 1.92x on distribution metrics, 1.14x on common-sense metrics, and 1.62x on road compliance from adversarial benchmarks. 

**Abstract (ZH)**: 基于路径扩散的多样化和现实交通场景模拟 

---
# PROFusion: Robust and Accurate Dense Reconstruction via Camera Pose Regression and Optimization 

**Title (ZH)**: PROFusion: 基于相机姿态回归与优化的鲁棒且准确的密集重建 

**Authors**: Siyan Dong, Zijun Wang, Lulu Cai, Yi Ma, Yanchao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.24236)  

**Abstract**: Real-time dense scene reconstruction during unstable camera motions is crucial for robotics, yet current RGB-D SLAM systems fail when cameras experience large viewpoint changes, fast motions, or sudden shaking. Classical optimization-based methods deliver high accuracy but fail with poor initialization during large motions, while learning-based approaches provide robustness but lack sufficient accuracy for dense reconstruction. We address this challenge through a combination of learning-based initialization with optimization-based refinement. Our method employs a camera pose regression network to predict metric-aware relative poses from consecutive RGB-D frames, which serve as reliable starting points for a randomized optimization algorithm that further aligns depth images with the scene geometry. Extensive experiments demonstrate promising results: our approach outperforms the best competitor on challenging benchmarks, while maintaining comparable accuracy on stable motion sequences. The system operates in real-time, showcasing that combining simple and principled techniques can achieve both robustness for unstable motions and accuracy for dense reconstruction. Project page: this https URL. 

**Abstract (ZH)**: 实时不稳定相机运动下的密集场景重建对于机器人技术至关重要，而当前的RGB-D SLAM系统在相机经历大幅度视角变化、快速运动或突然震动时会失效。基于经典优化的方法在大规模运动时因初始条件较差而无法提供高精度，而基于学习的方法虽然更具鲁棒性，但在密集重建方面缺乏足够的准确性。我们通过结合基于学习的初始化与基于优化的精修来应对这一挑战。我们的方法利用摄像头姿态回归网络从连续的RGB-D帧中预测出具备度量感知的相对姿态，作为随机优化算法的可靠起始点，进一步将深度图像与场景几何对齐。实验结果表明，我们的方法在具有挑战性的基准测试中优于现有最佳方法，同时在稳定运动序列上保持了相当的精度。系统实现了实时运行，展示了简单而原理明确的技术组合能够同时实现不稳定运动的鲁棒性和密集重建的准确性。项目页面: [这里](this https URL)。 

---
# BOSfM: A View Planning Framework for Optimal 3D Reconstruction of Agricultural Scenes 

**Title (ZH)**: BOSfM: 农业场景最佳3D重建的视角规划框架 

**Authors**: Athanasios Bacharis, Konstantinos D. Polyzos, Georgios B. Giannakis, Nikolaos Papanikolopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2509.24126)  

**Abstract**: Active vision (AV) has been in the spotlight of robotics research due to its emergence in numerous applications including agricultural tasks such as precision crop monitoring and autonomous harvesting to list a few. A major AV problem that gained popularity is the 3D reconstruction of targeted environments using 2D images from diverse viewpoints. While collecting and processing a large number of arbitrarily captured 2D images can be arduous in many practical scenarios, a more efficient solution involves optimizing the placement of available cameras in 3D space to capture fewer, yet more informative, images that provide sufficient visual information for effective reconstruction of the environment of interest. This process termed as view planning (VP), can be markedly challenged (i) by noise emerging in the location of the cameras and/or in the extracted images, and (ii) by the need to generalize well in other unknown similar agricultural environments without need for re-optimizing or re-training. To cope with these challenges, the present work presents a novel VP framework that considers a reconstruction quality-based optimization formulation that relies on the notion of `structure-from-motion' to reconstruct the 3D structure of the sought environment from the selected 2D images. With no analytic expression of the optimization function and with costly function evaluations, a Bayesian optimization approach is proposed to efficiently carry out the VP process using only a few function evaluations, while accounting for different noise cases. Numerical tests on both simulated and real agricultural settings signify the benefits of the advocated VP approach in efficiently estimating the optimal camera placement to accurately reconstruct 3D environments of interest, and generalize well on similar unknown environments. 

**Abstract (ZH)**: 基于运动结构的农业环境主动视图规划方法 

---
# Prepare for Warp Speed: Sub-millisecond Visual Place Recognition Using Event Cameras 

**Title (ZH)**: 准备高速模式：使用事件相机的亚毫秒级视觉.place识别 

**Authors**: Vignesh Ramanathan, Michael Milford, Tobias Fischer  

**Link**: [PDF](https://arxiv.org/pdf/2509.24094)  

**Abstract**: Visual Place Recognition (VPR) enables systems to identify previously visited locations within a map, a fundamental task for autonomous navigation. Prior works have developed VPR solutions using event cameras, which asynchronously measure per-pixel brightness changes with microsecond temporal resolution. However, these approaches rely on dense representations of the inherently sparse camera output and require tens to hundreds of milliseconds of event data to predict a place. Here, we break this paradigm with Flash, a lightweight VPR system that predicts places using sub-millisecond slices of event data. Our method is based on the observation that active pixel locations provide strong discriminative features for VPR. Flash encodes these active pixel locations using efficient binary frames and computes similarities via fast bitwise operations, which are then normalized based on the relative event activity in the query and reference frames. Flash improves Recall@1 for sub-millisecond VPR over existing baselines by 11.33x on the indoor QCR-Event-Dataset and 5.92x on the 8 km Brisbane-Event-VPR dataset. Moreover, our approach reduces the duration for which the robot must operate without awareness of its position, as evidenced by a localization latency metric we term Time to Correct Match (TCM). To the best of our knowledge, this is the first work to demonstrate sub-millisecond VPR using event cameras. 

**Abstract (ZH)**: 基于事件的极短时视觉地点识别（VPR）abling系统在地图中识别先前访问的位置，是自主导航的基本任务。以往研究利用事件摄像头开发VPR解决方案，这些摄像头以微秒级时间分辨率异步测量每個像素的亮度变化。然而，这些方法依赖于密集表示的固有稀疏摄像头输出，并需要数十到数百毫秒的事件数据来预测地点。在这里，我们通过 flash，一种轻量级VPR系统打破这一范式，该系统使用极短时事件数据片段预测地点。我们的方法基于观察到活跃像素位置为VPR提供了强烈的鉴别特征。Flash使用高效的二进制帧编码这些活跃像素位置，并通过快速位操作计算相似性，然后基于查询帧和参考帧的相对事件活动进行归一化。Flash在室内的QCR-Event-Dataset上将Recall@1的VPR性能比现有基线提高11.33倍，在8公里的布里斯班-事件-VPR数据集上提高5.92倍。此外，我们的方法减少了机器人在无位置意识状态下操作的持续时间，如我们所称的时间到正确匹配（TCM）的本地化延迟度量所示。据我们所知，这是首次使用事件摄像头实现极短时VPR的工作。 

---
# GLUE: Global-Local Unified Encoding for Imitation Learning via Key-Patch Tracking 

**Title (ZH)**: GLUE: 全局-局部统一编码在关键patches跟踪下的 imitation 学习 

**Authors**: Ye Chen, Zichen Zhou, Jianyu Dou, Te Cui, Yi Yang, Yufeng Yue  

**Link**: [PDF](https://arxiv.org/pdf/2509.23220)  

**Abstract**: In recent years, visual representation learning has gained widespread attention in robotic imitation learning. However, in complex Out-of-Distribution(OOD) settings characterized by clutter and occlusion, the attention of global visual representations can be diluted or interfered, leading to degraded policy performance. The invariance of local representations for task-relevant objects offers a solution. By efficiently utilizing these local representations, training and testing data can be mapped to a more similar feature space, thereby mitigating the covariate shift problem. Accordingly, we propose GLUE, a global-local unified encoding framework for imitation learning based on key-patch tracking. GLUE selects and tracks key-patches as critical local representations by employing a text-guided mechanism. It features a novel fusion framework where global patch features query local patches to distill essential information, yielding fine-grained local features with low heterogeneity relative to the global context. This fused representation steers the robot's visual attention toward task-relevant objects and preserves precise global context, which together align the training and testing distributions into a similar and task-informative feature space, ultimately enhancing the robustness of the imitation learning policy. Experiments demonstrate that GLUE achieves strong performance across diverse tasks in both simulation and real-world settings, outperforming the strongest baseline by 17.6% in simulation, 36.3% in real-world environments, and 58.3% on real-world generalization settings. The project website of GLUE is available at this https URL. 

**Abstract (ZH)**: 近年来，视觉表征学习在机器人模仿学习中受到了广泛关注。然而，在由杂乱和遮挡特征的复杂Out-of-Distribution(OOD)设置中，全局视觉表征的关注度可能会被稀释或干扰，导致政策性能下降。任务相关信息对象的局部表征不变性提供了一种解决方案。通过有效地利用这些局部表征，训练和测试数据可以映射到更相似的特征空间，从而缓解协变量转移问题。据此，我们提出了一种基于关键片段跟踪的全局-局部统一编码框架GLUE。GLUE通过采用文本引导机制选择和跟踪关键片段，作为关键局部表征。它具有一个新颖的融合框架，其中全局片段特征查询局部片段以提炼关键信息，生成相对于全局上下文低异质性的精细局部特征。这种融合表示引导机器人将视觉注意力集中于任务相关信息对象，并保留精确的全局上下文，从而使训练和测试分布对齐到一个相似且具有任务信息的特征空间，最终增强模仿学习策略的鲁棒性。实验结果表明，GLUE在模拟和现实世界设置中的多种任务上表现出色，在模拟环境中比最强基线高出17.6%，在现实世界环境中高出36.3%，在现实世界泛化设置上高出58.3%。GLUE项目的官方网站可通过该网址访问。 

---
# Persistent Autoregressive Mapping with Traffic Rules for Autonomous Driving 

**Title (ZH)**: 持续自回归映射结合交通规则的自动驾驶技术 

**Authors**: Shiyi Liang, Xinyuan Chang, Changjie Wu, Huiyuan Yan, Yifan Bai, Xinran Liu, Hang Zhang, Yujian Yuan, Shuang Zeng, Mu Xu, Xing Wei  

**Link**: [PDF](https://arxiv.org/pdf/2509.22756)  

**Abstract**: Safe autonomous driving requires both accurate HD map construction and persistent awareness of traffic rules, even when their associated signs are no longer visible. However, existing methods either focus solely on geometric elements or treat rules as temporary classifications, failing to capture their persistent effectiveness across extended driving sequences. In this paper, we present PAMR (Persistent Autoregressive Mapping with Traffic Rules), a novel framework that performs autoregressive co-construction of lane vectors and traffic rules from visual observations. Our approach introduces two key mechanisms: Map-Rule Co-Construction for processing driving scenes in temporal segments, and Map-Rule Cache for maintaining rule consistency across these segments. To properly evaluate continuous and consistent map generation, we develop MapDRv2, featuring improved lane geometry annotations. Extensive experiments demonstrate that PAMR achieves superior performance in joint vector-rule mapping tasks, while maintaining persistent rule effectiveness throughout extended driving sequences. 

**Abstract (ZH)**: 持久性车道向量与交通规则联合构建的自回归映射（PAMR） 

---
# Self-driving cars: Are we there yet? 

**Title (ZH)**: 自动驾驶汽车：我们到了吗？ 

**Authors**: Merve Atasever, Zhuochen Liu, Qingpei Li, Akshay Hitendra Shah, Hans Walker, Jyotirmoy V. Deshmukh, Rahul Jain  

**Link**: [PDF](https://arxiv.org/pdf/2509.22754)  

**Abstract**: Autonomous driving remains a highly active research domain that seeks to enable vehicles to perceive dynamic environments, predict the future trajectories of traffic agents such as vehicles, pedestrians, and cyclists and plan safe and efficient future motions. To advance the field, several competitive platforms and benchmarks have been established to provide standardized datasets and evaluation protocols. Among these, leaderboards by the CARLA organization and nuPlan and the Waymo Open Dataset have become leading benchmarks for assessing motion planning algorithms. Each offers a unique dataset and challenging planning problems spanning a wide range of driving scenarios and conditions. In this study, we present a comprehensive comparative analysis of the motion planning methods featured on these three leaderboards. To ensure a fair and unified evaluation, we adopt CARLA leaderboard v2.0 as our common evaluation platform and modify the selected models for compatibility. By highlighting the strengths and weaknesses of current approaches, we identify prevailing trends, common challenges, and suggest potential directions for advancing motion planning research. 

**Abstract (ZH)**: 自主泊车 remains 一个高度活跃的研究领域，旨在使车辆能够感知动态环境、预测交通参与者的未来轨迹（如车辆、行人和骑自行车者），并规划安全高效的未来行动。为了推进该领域的发展，已经建立了多个竞争平台和基准，提供了标准化的数据集和评估协议。在这些平台中，CARLA组织的排行榜、nuPlan以及Waymo开放数据集已成为评估运动规划算法的主要基准。每个平台都提供了独特数据集和广泛驾驶场景下的具有挑战性的规划问题。在本研究中，我们对这三个排行榜上的运动规划方法进行了全面的比较分析。为了确保公平统一的评估，我们采用CARLA排行榜v2.0作为共同评估平台，并对所选模型进行修改以实现兼容性。通过突出当前方法的优点和不足，我们识别了现有趋势、普遍挑战，并建议了促进运动规划研究进展的潜在方向。 

---
# Fast Feature Field ($\text{F}^3$): A Predictive Representation of Events 

**Title (ZH)**: 快速特征场 ($\text{F}^3$): 事件的预测表示 

**Authors**: Richeek Das, Kostas Daniilidis, Pratik Chaudhari  

**Link**: [PDF](https://arxiv.org/pdf/2509.25146)  

**Abstract**: This paper develops a mathematical argument and algorithms for building representations of data from event-based cameras, that we call Fast Feature Field ($\text{F}^3$). We learn this representation by predicting future events from past events and show that it preserves scene structure and motion information. $\text{F}^3$ exploits the sparsity of event data and is robust to noise and variations in event rates. It can be computed efficiently using ideas from multi-resolution hash encoding and deep sets - achieving 120 Hz at HD and 440 Hz at VGA resolutions. $\text{F}^3$ represents events within a contiguous spatiotemporal volume as a multi-channel image, enabling a range of downstream tasks. We obtain state-of-the-art performance on optical flow estimation, semantic segmentation, and monocular metric depth estimation, on data from three robotic platforms (a car, a quadruped robot and a flying platform), across different lighting conditions (daytime, nighttime), environments (indoors, outdoors, urban, as well as off-road) and dynamic vision sensors (resolutions and event rates). Our implementations can predict these tasks at 25-75 Hz at HD resolution. 

**Abstract (ZH)**: 本文开发了一种用于事件驱动摄像头数据表示的数学论证和算法，我们称之为快速特征场（$\text{F}^3$）。通过从过去事件预测未来事件来学习这种表示，并展示了其能够保留场景结构和运动信息。$\text{F}^3$ 利用事件数据的稀疏性，对噪声和事件率变化具有鲁棒性。其高效计算利用了多分辨率哈希编码和深度集合的理念，在高清分辨率下达到120 Hz，在VGA分辨率下达到440 Hz。$\text{F}^3$ 将事件表示为连续的空间-时间体积中的多通道图像，便于执行一系列下游任务。我们在三款机器人平台（一辆汽车、四足机器人和飞行平台）的数据上，在不同光照条件（白天、夜晚）、不同环境（室内、室外、城市以及非铺装道路）和不同动态视觉传感器（分辨率和事件率）下，均取得了最优性能。我们的实现可以在高清分辨率下以25-75 Hz的速度预测这些任务。 

---
# When Autonomous Vehicle Meets V2X Cooperative Perception: How Far Are We? 

**Title (ZH)**: 当自动驾驶车辆遇到V2X协同感知：我们还有多远？ 

**Authors**: An Guo, Shuoxiao Zhang, Enyi Tang, Xinyu Gao, Haomin Pang, Haoxiang Tian, Yanzhou Mu, Wu Wen, Chunrong Fang, Zhenyu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.24927)  

**Abstract**: With the tremendous advancement of deep learning and communication technology, Vehicle-to-Everything (V2X) cooperative perception has the potential to address limitations in sensing distant objects and occlusion for a single-agent perception system. V2X cooperative perception systems are software systems characterized by diverse sensor types and cooperative agents, varying fusion schemes, and operation under different communication conditions. Therefore, their complex composition gives rise to numerous operational challenges. Furthermore, when cooperative perception systems produce erroneous predictions, the types of errors and their underlying causes remain insufficiently explored. To bridge this gap, we take an initial step by conducting an empirical study of V2X cooperative perception. To systematically evaluate the impact of cooperative perception on the ego vehicle's perception performance, we identify and analyze six prevalent error patterns in cooperative perception systems. We further conduct a systematic evaluation of the critical components of these systems through our large-scale study and identify the following key findings: (1) The LiDAR-based cooperation configuration exhibits the highest perception performance; (2) Vehicle-to-infrastructure (V2I) and vehicle-to-vehicle (V2V) communication exhibit distinct cooperative perception performance under different fusion schemes; (3) Increased cooperative perception errors may result in a higher frequency of driving violations; (4) Cooperative perception systems are not robust against communication interference when running online. Our results reveal potential risks and vulnerabilities in critical components of cooperative perception systems. We hope that our findings can better promote the design and repair of cooperative perception systems. 

**Abstract (ZH)**: 随着深度学习和通信技术的飞速发展， Vehicle-to-Everything (V2X) 合作感知具有解决单Agent感知系统在探测远距离物体和遮挡方面局限性的潜力。V2X 合作感知系统是由多种传感器类型和合作代理、不同的融合方案以及在不同通信条件下运作的软件系统特征化。因此，其复杂的组成产生了众多的操作挑战。此外，当合作感知系统产生错误预测时，错误类型及其根本原因仍缺乏充分探索。为弥补这一空白，我们通过 empirical 研究初步探索 V2X 合作感知。为了系统评估合作感知对自主车辆感知性能的影响，我们识别并分析了合作感知系统中六种常见的错误模式。进一步通过大规模研究系统评估这些系统的关键组件，并得出以下关键发现：(1) 基于LiDAR的合作配置表现出最高的感知性能；(2) 车辆到基础设施（V2I）和车辆到车辆（V2V）通信在不同的融合方案下表现出不同的合作感知性能；(3) 合作感知错误的增加可能导致驾驶违规频率的提高；(4) 在线运行时，合作感知系统对通信干扰不够 robust。研究结果揭示了合作感知系统关键组件中的潜在风险和脆弱性。我们希望我们的发现能更好地促进合作感知系统的规划设计和修复。 

---
# ThermalGen: Style-Disentangled Flow-Based Generative Models for RGB-to-Thermal Image Translation 

**Title (ZH)**: ThermalGen：基于流的风格解耦生成模型用于RGB到热图图像转换 

**Authors**: Jiuhong Xiao, Roshan Nayak, Ning Zhang, Daniel Tortei, Giuseppe Loianno  

**Link**: [PDF](https://arxiv.org/pdf/2509.24878)  

**Abstract**: Paired RGB-thermal data is crucial for visual-thermal sensor fusion and cross-modality tasks, including important applications such as multi-modal image alignment and retrieval. However, the scarcity of synchronized and calibrated RGB-thermal image pairs presents a major obstacle to progress in these areas. To overcome this challenge, RGB-to-Thermal (RGB-T) image translation has emerged as a promising solution, enabling the synthesis of thermal images from abundant RGB datasets for training purposes. In this study, we propose ThermalGen, an adaptive flow-based generative model for RGB-T image translation, incorporating an RGB image conditioning architecture and a style-disentangled mechanism. To support large-scale training, we curated eight public satellite-aerial, aerial, and ground RGB-T paired datasets, and introduced three new large-scale satellite-aerial RGB-T datasets--DJI-day, Bosonplus-day, and Bosonplus-night--captured across diverse times, sensor types, and geographic regions. Extensive evaluations across multiple RGB-T benchmarks demonstrate that ThermalGen achieves comparable or superior translation performance compared to existing GAN-based and diffusion-based methods. To our knowledge, ThermalGen is the first RGB-T image translation model capable of synthesizing thermal images that reflect significant variations in viewpoints, sensor characteristics, and environmental conditions. Project page: this http URL 

**Abstract (ZH)**: 配对的RGB-热成像数据对于视觉-热传感器融合及跨模态任务至关重要，包括多模态图像对齐和检索等重要应用。然而，同步和校准的RGB-热成像配对数据的稀缺性极大地阻碍了这些领域的进展。为了克服这一挑战，RGB到热成像（RGB-T）图像转换已成为一种有前景的解决方案，使人们能够从丰富的RGB数据集中合成热图像以供训练使用。在本研究中，我们提出了ThermalGen，这是一种适应性的基于流的生成模型，用于RGB-T图像转换，结合了RGB图像条件化架构和风格解耦机制。为支持大规模训练，我们汇聚了八个公开的卫星-航空、航空和地面RGB-T配对数据集，并引入了三个新的大型卫星-航空RGB-T数据集——DJI-day、Bosonplus-day和Bosonplus-night，这些数据集在不同的时间、传感器类型和地理区域进行了拍摄。在多个RGB-T基准上的广泛评估表明，ThermalGen在转换性能上达到了与现有GAN基和扩散基方法相当或更优的水平。据我们所知，ThermalGen是首款能够合成反映显著视角变化、传感器特性和环境条件差异的热图像的RGB-T图像转换模型。 

---
# Evaluation of Polarimetric Fusion for Semantic Segmentation in Aquatic Environments 

**Title (ZH)**: 水文环境中极化融合的语义分割评价 

**Authors**: Luis F. W. Batista, Tom Bourbon, Cedric Pradalier  

**Link**: [PDF](https://arxiv.org/pdf/2509.24731)  

**Abstract**: Accurate segmentation of floating debris on water is often compromised by surface glare and changing outdoor illumination. Polarimetric imaging offers a single-sensor route to mitigate water-surface glare that disrupts semantic segmentation of floating objects. We benchmark state-of-the-art fusion networks on PoTATO, a public dataset of polarimetric images of plastic bottles in inland waterways, and compare their performance with single-image baselines using traditional models. Our results indicate that polarimetric cues help recover low-contrast objects and suppress reflection-induced false positives, raising mean IoU and lowering contour error relative to RGB inputs. These sharper masks come at a cost: the additional channels enlarge the models increasing the computational load and introducing the risk of new false positives. By providing a reproducible, diagnostic benchmark and publicly available code, we hope to help researchers choose if polarized cameras are suitable for their applications and to accelerate related research. 

**Abstract (ZH)**: 水面上漂浮垃圾的准确分割常受到水面眩光和变化户外光照的干扰。偏振成像提供了一种通过单传感器方式减轻影响漂浮物语义分割的水面眩光的方法。我们在公开的数据集PoTATO上对最先进的融合网络进行基准测试，并将其性能与使用传统模型的单图像基线进行比较。我们的结果显示，偏振线索有助于恢复低对比度物体并抑制反射引起的假阳性，相对RGB输入提高了平均IoU并降低了轮廓误差。然而，这些更清晰的掩码也存在成本：额外的通道增大了模型，增加了计算负担并引入了新的假阳性风险。通过提供可重复的诊断基准和公开代码，我们希望帮助研究人员判断偏振相机是否适合其应用，并加速相关研究。 

---
# SCOPE: Semantic Conditioning for Sim2Real Category-Level Object Pose Estimation in Robotics 

**Title (ZH)**: 语义条件化在机器人领域中的Sim2Real类别级物体姿态估计 

**Authors**: Peter Hönig, Stefan Thalhammer, Jean-Baptiste Weibel, Matthias Hirschmanner, Markus Vincze  

**Link**: [PDF](https://arxiv.org/pdf/2509.24572)  

**Abstract**: Object manipulation requires accurate object pose estimation. In open environments, robots encounter unknown objects, which requires semantic understanding in order to generalize both to known categories and beyond. To resolve this challenge, we present SCOPE, a diffusion-based category-level object pose estimation model that eliminates the need for discrete category labels by leveraging DINOv2 features as continuous semantic priors. By combining these DINOv2 features with photorealistic training data and a noise model for point normals, we reduce the Sim2Real gap in category-level object pose estimation. Furthermore, injecting the continuous semantic priors via cross-attention enables SCOPE to learn canonicalized object coordinate systems across object instances beyond the distribution of known categories. SCOPE outperforms the current state of the art in synthetically trained category-level object pose estimation, achieving a relative improvement of 31.9\% on the 5$^\circ$5cm metric. Additional experiments on two instance-level datasets demonstrate generalization beyond known object categories, enabling grasping of unseen objects from unknown categories with a success rate of up to 100\%. Code available: this https URL. 

**Abstract (ZH)**: 物体操作需要准确的物体姿态估计。在开放环境中，机器人遇到未知物体，这要求进行语义理解以在已知类别和未知类别之间进行泛化。为了解决这一挑战，我们提出了SCOPE，这是一种基于扩散的类别级物体姿态估计模型，通过利用DINOv2特征作为连续的语义先验来消除对离散类别标签的需求。通过将这些DINOv2特征与照片真实感的训练数据和点法线噪声模型相结合，我们缩小了类别级物体姿态估计中的Sim2Real差距。此外，通过交叉注意力注入连续的语义先验使SCOPE能够在已知类别分布之外学习标准化的物体坐标系统。SCOPE在合成训练的类别级物体姿态估计中超越了当前最佳方法，在5°5cm指标上取得了31.9%的相对改进。在两个实例级数据集上的附加实验进一步证明了其在已知物体类别之外的泛化能力，从而使机器人能够从未知类别中抓取未见过的物体，成功率可达100%。代码可获取：this https URL。 

---
# FreeAction: Training-Free Techniques for Enhanced Fidelity of Trajectory-to-Video Generation 

**Title (ZH)**: FreeAction: 无需训练的技术以提高轨迹到视频生成保真度 

**Authors**: Seungwook Kim, Seunghyeon Lee, Minsu Cho  

**Link**: [PDF](https://arxiv.org/pdf/2509.24241)  

**Abstract**: Generating realistic robot videos from explicit action trajectories is a critical step toward building effective world models and robotics foundation models. We introduce two training-free, inference-time techniques that fully exploit explicit action parameters in diffusion-based robot video generation. Instead of treating action vectors as passive conditioning signals, our methods actively incorporate them to guide both the classifier-free guidance process and the initialization of Gaussian latents. First, action-scaled classifier-free guidance dynamically modulates guidance strength in proportion to action magnitude, enhancing controllability over motion intensity. Second, action-scaled noise truncation adjusts the distribution of initially sampled noise to better align with the desired motion dynamics. Experiments on real robot manipulation datasets demonstrate that these techniques significantly improve action coherence and visual quality across diverse robot environments. 

**Abstract (ZH)**: 从显式动作轨迹生成真實机器人视频是建立有效世界模型和机器人基础模型的关键步骤。我们介绍两种无需训练、在推理时使用的技巧，充分利用基于扩散的机器人视频生成中的显式动作参数。我们的方法不是将动作向量视为被动的条件信号，而是主动将其整合，以指导无分类器引导过程并初始化高斯潜在变量。首先，动作缩放的无分类器引导动态按动作幅度调整引导强度，增强对运动强度的可控性。其次，动作缩放的噪声截断调整初始采样噪声的分布，以更好地与期望的运动动力学对齐。实验表明，这些技术在多种机器人环境中显著提高了动作连贯性和视觉质量。 

---
# GRS-SLAM3R: Real-Time Dense SLAM with Gated Recurrent State 

**Title (ZH)**: GRS-SLAM3R: 基于门控循环状态的实时密集SLAM 

**Authors**: Guole Shen, Tianchen Deng, Yanbo Wang, Yongtao Chen, Yilin Shen, Jiuming Liu, Jingchuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23737)  

**Abstract**: DUSt3R-based end-to-end scene reconstruction has recently shown promising results in dense visual SLAM. However, most existing methods only use image pairs to estimate pointmaps, overlooking spatial memory and global this http URL this end, we introduce GRS-SLAM3R, an end-to-end SLAM framework for dense scene reconstruction and pose estimation from RGB images without any prior knowledge of the scene or camera parameters. Unlike existing DUSt3R-based frameworks, which operate on all image pairs and predict per-pair point maps in local coordinate frames, our method supports sequentialized input and incrementally estimates metric-scale point clouds in the global coordinate. In order to improve consistent spatial correlation, we use a latent state for spatial memory and design a transformer-based gated update module to reset and update the spatial memory that continuously aggregates and tracks relevant 3D information across frames. Furthermore, we partition the scene into submaps, apply local alignment within each submap, and register all submaps into a common world frame using relative constraints, producing a globally consistent map. Experiments on various datasets show that our framework achieves superior reconstruction accuracy while maintaining real-time performance. 

**Abstract (ZH)**: 基于DUSt3R的端到端场景重建在稠密视觉SLAM中取得了令人 promising 的结果。然而，现有的大多数方法仅使用图像对来估计点图，忽略了空间记忆和全局信息。为此，我们提出了GRS-SLAM3R，这是一种基于RGB图像进行稠密场景重建和姿态估计的端到端SLAM框架，无需任何场景或相机参数先验知识。与现有的基于DUSt3R的框架不同，后者在所有图像对上操作并预测局部坐标系中的点图，我们的方法支持顺序输入，并在全局坐标系中增量地估计米尺度点云。为了提高一致的空间相关性，我们使用隐状态进行空间记忆，并设计了一个变压器基门控更新模块，以重置和更新连续聚合和跟踪各帧相关3D信息的空间记忆。此外，我们将场景划分为子图，对每个子图内部进行局部对齐，并使用相对约束将所有子图注册到一个共同的世界坐标系中，生成全局一致的地图。在各种数据集上的实验表明，我们的框架在保持实时性能的同时实现了更优的重建精度。 

---
# FastViDAR: Real-Time Omnidirectional Depth Estimation via Alternative Hierarchical Attention 

**Title (ZH)**: FastViDAR：基于交替分层注意力的实时全景深度估计 

**Authors**: Hangtian Zhao, Xiang Chen, Yizhe Li, Qianhao Wang, Haibo Lu, Fei Gao  

**Link**: [PDF](https://arxiv.org/pdf/2509.23733)  

**Abstract**: In this paper we propose FastViDAR, a novel framework that takes four fisheye camera inputs and produces a full $360^\circ$ depth map along with per-camera depth, fusion depth, and confidence estimates. Our main contributions are: (1) We introduce Alternative Hierarchical Attention (AHA) mechanism that efficiently fuses features across views through separate intra-frame and inter-frame windowed self-attention, achieving cross-view feature mixing with reduced overhead. (2) We propose a novel ERP fusion approach that projects multi-view depth estimates to a shared equirectangular coordinate system to obtain the final fusion depth. (3) We generate ERP image-depth pairs using HM3D and 2D3D-S datasets for comprehensive evaluation, demonstrating competitive zero-shot performance on real datasets while achieving up to 20 FPS on NVIDIA Orin NX embedded hardware. Project page: \href{this https URL}{this https URL} 

**Abstract (ZH)**: 本文提出FastViDAR，一种新型框架，利用四个鱼眼摄像头输入生成全视角360°深度图以及每个摄像头的深度、融合深度和置信度估计。我们的主要贡献包括：(1) 引入了替代分层注意力（AHA）机制，通过单独的帧内和帧间窗口自注意力高效融合视图间特征，实现跨视图特征混合并减少开销。(2) 提出了一种新颖的ERP融合方法，将多视角深度估计投影到共享的等角坐标系中以获得最终的融合深度。(3) 使用HM3D和2D3D-S数据集生成ERP图像-深度配对以进行全面评估，展示了在真实数据集上具有竞争力的零样本性能，并在NVIDIA Orin NX嵌入式硬件上实现高达20 FPS。项目页面：[此链接] 

---
# Color-Pair Guided Robust Zero-Shot 6D Pose Estimation and Tracking of Cluttered Objects on Edge Devices 

**Title (ZH)**: 颜色配对引导的鲁棒零样本6D姿态估计与杂乱环境中对象的跟踪 边缘设备 

**Authors**: Xingjian Yang, Ashis G. Banerjee  

**Link**: [PDF](https://arxiv.org/pdf/2509.23647)  

**Abstract**: Robust 6D pose estimation of novel objects under challenging illumination remains a significant challenge, often requiring a trade-off between accurate initial pose estimation and efficient real-time tracking. We present a unified framework explicitly designed for efficient execution on edge devices, which synergizes a robust initial estimation module with a fast motion-based tracker. The key to our approach is a shared, lighting-invariant color-pair feature representation that forms a consistent foundation for both stages. For initial estimation, this feature facilitates robust registration between the live RGB-D view and the object's 3D mesh. For tracking, the same feature logic validates temporal correspondences, enabling a lightweight model to reliably regress the object's motion. Extensive experiments on benchmark datasets demonstrate that our integrated approach is both effective and robust, providing competitive pose estimation accuracy while maintaining high-fidelity tracking even through abrupt pose changes. 

**Abstract (ZH)**: 在挑战性光照下鲁棒的新型对象6D姿态估计仍然是一项重大挑战，往往需要在准确的初始姿态估计和高效的实时跟踪之间做出权衡。我们提出了一种统一框架，专门设计用于边缘设备上的高效执行，该框架将稳健的初始估计模块与快速运动跟踪器相结合。我们方法的核心是一种共享的、光照不变的颜色对特征表示，为两个阶段提供了一致的基础。在初始估计阶段，该特征使得活RGB-D视图与对象的3D网格之间的稳健配准成为可能。在跟踪阶段，相同的特征逻辑验证了时态对应关系，使轻量级模型能够可靠地回归对象的运动。在基准数据集上的广泛实验表明，我们集成的方法既是有效的又是鲁棒的，在通过突然的姿态变化时仍然能够保持高保真跟踪，并提供具有竞争力的姿态估计精度。 

---
# Motion Informed Needle Segmentation in Ultrasound Images 

**Title (ZH)**: 超声图像中的运动指导针头分割 

**Authors**: Raghavv Goel, Cecilia Morales, Manpreet Singh, Artur Dubrawski, John Galeotti, Howie Choset  

**Link**: [PDF](https://arxiv.org/pdf/2312.01239)  

**Abstract**: Segmenting a moving needle in ultrasound images is challenging due to the presence of artifacts, noise, and needle occlusion. This task becomes even more demanding in scenarios where data availability is limited. In this paper, we present a novel approach for needle segmentation for 2D ultrasound that combines classical Kalman Filter (KF) techniques with data-driven learning, incorporating both needle features and needle motion. Our method offers three key contributions. First, we propose a compatible framework that seamlessly integrates into commonly used encoder-decoder style architectures. Second, we demonstrate superior performance compared to recent state-of-the-art needle segmentation models using our novel convolutional neural network (CNN) based KF-inspired block, achieving a 15\% reduction in pixel-wise needle tip error and an 8\% reduction in length error. Third, to our knowledge we are the first to implement a learnable filter to incorporate non-linear needle motion for improving needle segmentation. 

**Abstract (ZH)**: 基于经典的卡尔曼滤波器技术和数据驱动学习的2D超声针段化方法 

---
# Mixture-of-Visual-Thoughts: Exploring Context-Adaptive Reasoning Mode Selection for General Visual Reasoning 

**Title (ZH)**: 多模态视觉思考：探索面向上下文的推理模式选择的通用视觉推理 

**Authors**: Zejun Li, Yingxiu Zhao, Jiwen Zhang, Siyuan Wang, Yang Yao, Runzhou Zhao, Jun Song, Bo Zheng, Zhongyu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2509.22746)  

**Abstract**: Current visual reasoning methods mainly focus on exploring specific reasoning modes. Although improvements can be achieved in particular domains, they struggle to develop general reasoning capabilities. Inspired by this, we propose a novel adaptive reasoning paradigm, Mixture-of-Visual-Thoughts (MoVT), which unifies different reasoning modes within a single model and guides it to select the appropriate mode based on context. To achieve this, we introduce AdaVaR, a two-stage Adaptive Visual Reasoning learning framework: different modes are unified and learned during the supervised cold-start stage, and the mode selection capability is induced via an RL process with a carefully designed AdaGRPO algorithm. Extensive experiments show that AdaVaR effectively guides the model to learn and differentiate multiple modes and perform context-adaptive mode selection, achieving consistent improvement across various scenarios, highlighting MoVT as an effective solution for building general visual reasoning models. 

**Abstract (ZH)**: 当前的视觉推理方法主要集中在探索特定的推理模式。尽管可以在特定领域实现改进，但它们难以发展出通用的推理能力。受此启发，我们提出了一种新的自适应推理范式——混合视觉思考（MoVT），该范式在单一模型中统一了不同的推理模式，并根据上下文引导模型选择合适的模式。为此，我们引入了AdaVaR，这是一种两阶段自适应视觉推理学习框架：在监督冷启动阶段统一并学习不同的模式，并通过精心设计的AdaGRPO算法的RL过程诱导模式选择能力。广泛实验表明，AdaVaR有效地引导模型学习和区分多种模式，并进行上下文适应的模式选择，实现了各种场景中的一致改进，突显了MoVT作为构建通用视觉推理模型的有效解决方案。 

---
# DC-VideoGen: Efficient Video Generation with Deep Compression Video Autoencoder 

**Title (ZH)**: DC-VideoGen: 基于深度压缩视频自编码器的高效视频生成 

**Authors**: Junyu Chen, Wenkun He, Yuchao Gu, Yuyang Zhao, Jincheng Yu, Junsong Chen, Dongyun Zou, Yujun Lin, Zhekai Zhang, Muyang Li, Haocheng Xi, Ligeng Zhu, Enze Xie, Song Han, Han Cai  

**Link**: [PDF](https://arxiv.org/pdf/2509.25182)  

**Abstract**: We introduce DC-VideoGen, a post-training acceleration framework for efficient video generation. DC-VideoGen can be applied to any pre-trained video diffusion model, improving efficiency by adapting it to a deep compression latent space with lightweight fine-tuning. The framework builds on two key innovations: (i) a Deep Compression Video Autoencoder with a novel chunk-causal temporal design that achieves 32x/64x spatial and 4x temporal compression while preserving reconstruction quality and generalization to longer videos; and (ii) AE-Adapt-V, a robust adaptation strategy that enables rapid and stable transfer of pre-trained models into the new latent space. Adapting the pre-trained Wan-2.1-14B model with DC-VideoGen requires only 10 GPU days on the NVIDIA H100 GPU. The accelerated models achieve up to 14.8x lower inference latency than their base counterparts without compromising quality, and further enable 2160x3840 video generation on a single GPU. Code: this https URL. 

**Abstract (ZH)**: DC-VideoGen：一种用于高效视频生成的后训练加速框架 

---
# DC-Gen: Post-Training Diffusion Acceleration with Deeply Compressed Latent Space 

**Title (ZH)**: DC-Gen: 训练后扩散加速与深度压缩潜在空间 

**Authors**: Wenkun He, Yuchao Gu, Junyu Chen, Dongyun Zou, Yujun Lin, Zhekai Zhang, Haocheng Xi, Muyang Li, Ligeng Zhu, Jincheng Yu, Junsong Chen, Enze Xie, Song Han, Han Cai  

**Link**: [PDF](https://arxiv.org/pdf/2509.25180)  

**Abstract**: Existing text-to-image diffusion models excel at generating high-quality images, but face significant efficiency challenges when scaled to high resolutions, like 4K image generation. While previous research accelerates diffusion models in various aspects, it seldom handles the inherent redundancy within the latent space. To bridge this gap, this paper introduces DC-Gen, a general framework that accelerates text-to-image diffusion models by leveraging a deeply compressed latent space. Rather than a costly training-from-scratch approach, DC-Gen uses an efficient post-training pipeline to preserve the quality of the base model. A key challenge in this paradigm is the representation gap between the base model's latent space and a deeply compressed latent space, which can lead to instability during direct fine-tuning. To overcome this, DC-Gen first bridges the representation gap with a lightweight embedding alignment training. Once the latent embeddings are aligned, only a small amount of LoRA fine-tuning is needed to unlock the base model's inherent generation quality. We verify DC-Gen's effectiveness on SANA and FLUX.1-Krea. The resulting DC-Gen-SANA and DC-Gen-FLUX models achieve quality comparable to their base models but with a significant speedup. Specifically, DC-Gen-FLUX reduces the latency of 4K image generation by 53x on the NVIDIA H100 GPU. When combined with NVFP4 SVDQuant, DC-Gen-FLUX generates a 4K image in just 3.5 seconds on a single NVIDIA 5090 GPU, achieving a total latency reduction of 138x compared to the base FLUX.1-Krea model. Code: this https URL. 

**Abstract (ZH)**: 现有的文本到图像扩散模型在生成高质量图像方面表现出色，但在扩展到高分辨率（如4K图像生成）时面临显著的效率挑战。尽管先前的研究在各种方面加速了扩散模型，但它们很少处理潜在空间内的固有冗余。为解决这一问题，本文提出了DC-Gen，这是一个通过利用深度压缩的潜在空间来加速文本到图像扩散模型的通用框架。DC-Gen 不采用从头训练的成本高昂的方法，而是使用高效的后训练管道来保留基模型的质量。在这个范式中，基模型的潜在空间与深度压缩的潜在空间之间的表示差距是一个关键挑战，这可能导致直接微调时的不稳定。为克服这一问题，DC-Gen 首先通过轻量级嵌入对齐训练来弥合表示差距。一旦潜在嵌入对齐，只需少量LoRA微调即可释放基模型固有的生成质量。我们在SANA和FLUX.1-Krea上验证了DC-Gen的有效性。由此生成的DC-Gen-SANA和DC-Gen-FLUX模型在质量上与基模型相当，但具有显著的速度提升。具体来说，DC-Gen-FLUX在NVIDIA H100 GPU上将4K图像生成的延迟降低了53倍。结合NVFP4 SVDQuant后，DC-Gen-FLUX在单个NVIDIA 5090 GPU上生成4K图像仅需3.5秒，与基FLUX.1-Krea模型相比，总延迟降低138倍。代码：见这里。 

---
# GSM8K-V: Can Vision Language Models Solve Grade School Math Word Problems in Visual Contexts 

**Title (ZH)**: GSM8K-V：视觉语言模型能否解决视觉背景下的一年级数学文字题？ 

**Authors**: Fan Yuan, Yuchen Yan, Yifan Jiang, Haoran Zhao, Tao Feng, Jinyan Chen, Yanwei Lou, Wenqi Zhang, Yongliang Shen, Weiming Lu, Jun Xiao, Yueting Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2509.25160)  

**Abstract**: Vision language models (VLMs) achieve unified modeling of images and text, enabling them to accomplish complex real-world tasks through perception, planning, and reasoning. Among these tasks, reasoning is particularly representative, with mathematical reasoning serving as a prominent example. It highlights the high-level capability of VLMs to comprehend mathematical information in images and to perform sophisticated reasoning. Recently, numerous visual mathematical reasoning benchmarks have been proposed, but they are often restricted to geometry, lack coverage of math word problems, and rarely assess reasoning across multiple images. To address these gaps, we introduce GSM8K-V, a purely visual multi-image mathematical reasoning benchmark. GSM8K-V is built by systematically mapping each sample from the widely used text-based GSM8K into visual form. Through a carefully designed automated image-generation pipeline combined with meticulous human annotation, we curate 1,319 high-quality samples. We evaluate a wide range of open-source and closed-source models on GSM8K-V. Results show that although existing VLMs have nearly saturated performance on text-based GSM8K, there remains substantial room for improvement on GSM8K-V. For example, the best-performing model, Gemini-2.5-Pro, achieves 95.22% accuracy on GSM8K but only 46.93% on GSM8K-V. We conduct a comprehensive analysis of GSM8K-V, examining the limitations of current models as well as potential directions for improvement. GSM8K-V offers a new perspective on visual mathematical reasoning and establishes a benchmark to guide the development of more robust and generalizable VLMs. 

**Abstract (ZH)**: 视觉语言模型（VLMs）实现了图像与文本的统一建模，使其能够通过感知、规划和推理来完成复杂的现实世界任务。在这类任务中，推理尤为典型，其中数学推理是一种突出的例子。它突显了VLMs在理解图像中的数学信息和进行复杂推理方面的高级能力。近期，提出了许多视觉数学推理基准，但它们往往局限于几何领域，缺乏数学文字问题的覆盖，并且很少评估跨多张图像的推理能力。为弥补这些不足，我们引入了GSM8K-V，一个纯粹基于视觉的多图像数学推理基准。GSM8K-V通过系统地将广泛使用的文本基于基准GSM8K中的每个样本映射到视觉形式构建而成。通过精心设计的自动化图像生成管道与细致的人工注释相结合，我们策划了1,319个高质量样本。我们在GSM8K-V上评估了各种开源和闭源模型。结果表明，虽然现有的VLMs在文本基于的GSM8K上的性能几乎饱和，但在GSM8K-V上仍有很大的改进空间。例如，表现最佳的模型Gemini-2.5-Pro在GSM8K上的准确率为95.22%，但在GSM8K-V上的准确率仅为46.93%。我们对GSM8K-V进行了全面分析，探讨了当前模型的局限性和改进的潜在方向。GSM8K-V为视觉数学推理提供了新的视角，并建立了指导更稳健和泛化能力更强的VLMs发展的一个基准。 

---
# Score Distillation of Flow Matching Models 

**Title (ZH)**: 流匹配模型的评分蒸馏 

**Authors**: Mingyuan Zhou, Yi Gu, Huangjie Zheng, Liangchen Song, Guande He, Yizhe Zhang, Wenze Hu, Yinfei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.25127)  

**Abstract**: Diffusion models achieve high-quality image generation but are limited by slow iterative sampling. Distillation methods alleviate this by enabling one- or few-step generation. Flow matching, originally introduced as a distinct framework, has since been shown to be theoretically equivalent to diffusion under Gaussian assumptions, raising the question of whether distillation techniques such as score distillation transfer directly. We provide a simple derivation -- based on Bayes' rule and conditional expectations -- that unifies Gaussian diffusion and flow matching without relying on ODE/SDE formulations. Building on this view, we extend Score identity Distillation (SiD) to pretrained text-to-image flow-matching models, including SANA, SD3-Medium, SD3.5-Medium/Large, and FLUX.1-dev, all with DiT backbones. Experiments show that, with only modest flow-matching- and DiT-specific adjustments, SiD works out of the box across these models, in both data-free and data-aided settings, without requiring teacher finetuning or architectural changes. This provides the first systematic evidence that score distillation applies broadly to text-to-image flow matching models, resolving prior concerns about stability and soundness and unifying acceleration techniques across diffusion- and flow-based generators. We will make the PyTorch implementation publicly available. 

**Abstract (ZH)**: 扩散模型能够生成高质量的图像，但迭代采样速度较慢。蒸馏方法通过使生成过程减少至一或几步来解决这一问题。流匹配最初被引入为一个独立的框架，后来在高斯假定下被证明与扩散模型在理论上等价，这引发了这样的疑问：是否可以直接将蒸馏技术如得分蒸馏应用于流匹配模型。我们基于贝叶斯规则和条件期望提供了简单的数学推导，以不依赖于常微分方程/随机微分方程形式的方式统一了高斯扩散和流匹配。在此基础上，我们拓展了得分身份蒸馏（SiD）技术，应用于预训练的文本到图像流匹配模型，包括SANA、SD3-Medium、SD3.5-Medium/Large和FLUX.1-dev，所有模型均基于DiT架构。实验表明，通过仅进行适度的流匹配和DiT特定调整，SiD在这些模型中无需教师微调或架构更改即可直接应用，无论是无数据辅助还是有数据辅助设置。这一发现提供了首次系统的证据，表明得分蒸馏广泛适用于文本到图像流匹配模型，解决了此前关于稳定性和正确性的问题，并实现了扩散生成器和流生成器加速技术的一体化。我们将公开发布PyTorch实现。 

---
# UniLat3D: Geometry-Appearance Unified Latents for Single-Stage 3D Generation 

**Title (ZH)**: UniLat3D: 同时统一几何与外观的单阶段三维生成潜变量 

**Authors**: Guanjun Wu, Jiemin Fang, Chen Yang, Sikuang Li, Taoran Yi, Jia Lu, Zanwei Zhou, Jiazhong Cen, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Xinggang Wang, Qi Tian  

**Link**: [PDF](https://arxiv.org/pdf/2509.25079)  

**Abstract**: High-fidelity 3D asset generation is crucial for various industries. While recent 3D pretrained models show strong capability in producing realistic content, most are built upon diffusion models and follow a two-stage pipeline that first generates geometry and then synthesizes appearance. Such a decoupled design tends to produce geometry-texture misalignment and non-negligible cost. In this paper, we propose UniLat3D, a unified framework that encodes geometry and appearance in a single latent space, enabling direct single-stage generation. Our key contribution is a geometry-appearance Unified VAE, which compresses high-resolution sparse features into a compact latent representation -- UniLat. UniLat integrates structural and visual information into a dense low-resolution latent, which can be efficiently decoded into diverse 3D formats, e.g., 3D Gaussians and meshes. Based on this unified representation, we train a single flow-matching model to map Gaussian noise directly into UniLat, eliminating redundant stages. Trained solely on public datasets, UniLat3D produces high-quality 3D assets in seconds from a single image, achieving superior appearance fidelity and geometric quality. More demos \& code are available at this https URL 

**Abstract (ZH)**: 高保真3D资产生成对于多个行业至关重要。虽然近期的3D预训练模型在产生逼真内容方面表现出强大的能力，但大多数模型基于扩散模型构建，并遵循两阶段管道，首先生成几何结构，然后合成外观。这种解耦设计往往会产生几何结构与纹理不匹配，并伴随着较高的成本。在本文中，我们提出了一种统一框架UniLat3D，该框架将几何结构和外观编码到单个潜在空间中，使得可以直接进行单阶段生成。我们的主要贡献是一种几何结构-外观统一的VAE，它将高分辨率稀疏特征压缩成紧凑的潜在表示UniLat。UniLat将结构和视觉信息整合到密集的低分辨率潜在空间中，可以高效地解码为多种3D格式，例如3D高斯函数和网格。基于这种统一的表示，我们训练了一个单阶段流匹配模型，直接将高斯噪声映射到UniLat，从而消除了冗余阶段。仅通过公共数据集训练，UniLat3D可以从单张图像中在几秒钟内生成高质量的3D资产，实现卓越的外观保真度和几何质量。更多演示与代码请访问此网址。 

---
# BRIDGE - Building Reinforcement-Learning Depth-to-Image Data Generation Engine for Monocular Depth Estimation 

**Title (ZH)**: BRIDGE - 建立用于单目深度估计的强化学习深度到图像数据生成引擎 

**Authors**: Dingning Liu, Haoyu Guo, Jingyi Zhou, Tong He  

**Link**: [PDF](https://arxiv.org/pdf/2509.25077)  

**Abstract**: Monocular Depth Estimation (MDE) is a foundational task for computer vision. Traditional methods are limited by data scarcity and quality, hindering their robustness. To overcome this, we propose BRIDGE, an RL-optimized depth-to-image (D2I) generation framework that synthesizes over 20M realistic and geometrically accurate RGB images, each intrinsically paired with its ground truth depth, from diverse source depth maps. Then we train our depth estimation model on this dataset, employing a hybrid supervision strategy that integrates teacher pseudo-labels with ground truth depth for comprehensive and robust training. This innovative data generation and training paradigm enables BRIDGE to achieve breakthroughs in scale and domain diversity, consistently outperforming existing state-of-the-art approaches quantitatively and in complex scene detail capture, thereby fostering general and robust depth features. Code and models are available at this https URL. 

**Abstract (ZH)**: 单目深度估计（MDE）是计算机视觉中的基础任务。传统方法受限于数据的稀缺性和质量，影响其鲁棒性。为克服这一问题，我们提出BRIDGE，一种基于强化学习优化的深度图到图像（D2I）生成框架，该框架从多种来源的深度图中合成超过2000万张真实且几何准确的RGB图像，并且每张图像都与其地面 truth深度图内嵌配对。然后，我们使用一种混合监督策略训练我们的深度估计模型，该策略结合了教师伪标签与地面 truth深度图，以实现全面且稳健的训练。这一创新的数据生成和训练范式使BRIDGE在规模和领域多样性方面取得突破，定量和定性上均优于现有最先进的方法，从而促进通用且鲁棒的深度特征的生成。代码和模型可在以下网址获取。 

---
# Fast Real-Time Pipeline for Robust Arm Gesture Recognition 

**Title (ZH)**: 快速稳健的手臂手势识别实时处理管道 

**Authors**: Milán Zsolt Bagladi, László Gulyás, Gergő Szalay  

**Link**: [PDF](https://arxiv.org/pdf/2509.25042)  

**Abstract**: This paper presents a real-time pipeline for dynamic arm gesture recognition based on OpenPose keypoint estimation, keypoint normalization, and a recurrent neural network classifier. The 1 x 1 normalization scheme and two feature representations (coordinate- and angle-based) are presented for the pipeline. In addition, an efficient method to improve robustness against camera angle variations is also introduced by using artificially rotated training data. Experiments on a custom traffic-control gesture dataset demonstrate high accuracy across varying viewing angles and speeds. Finally, an approach to calculate the speed of the arm signal (if necessary) is also presented. 

**Abstract (ZH)**: 基于OpenPose关键点估计、关键点归一化和循环神经网络分类器的实时动态手臂手势识别流水线及其应用 

---
# CLASP: Adaptive Spectral Clustering for Unsupervised Per-Image Segmentation 

**Title (ZH)**: CLASP:自适应谱聚类用于无监督单图像分割 

**Authors**: Max Curie, Paulo da Costa  

**Link**: [PDF](https://arxiv.org/pdf/2509.25016)  

**Abstract**: We introduce CLASP (Clustering via Adaptive Spectral Processing), a lightweight framework for unsupervised image segmentation that operates without any labeled data or finetuning. CLASP first extracts per patch features using a self supervised ViT encoder (DINO); then, it builds an affinity matrix and applies spectral clustering. To avoid manual tuning, we select the segment count automatically with a eigengap silhouette search, and we sharpen the boundaries with a fully connected DenseCRF. Despite its simplicity and training free nature, CLASP attains competitive mIoU and pixel accuracy on COCO Stuff and ADE20K, matching recent unsupervised baselines. The zero training design makes CLASP a strong, easily reproducible baseline for large unannotated corpora especially common in digital advertising and marketing workflows such as brand safety screening, creative asset curation, and social media content moderation 

**Abstract (ZH)**: CLASP（自适应光谱处理聚类）：一种轻量级的无监督图像分割框架 

---
# Light-SQ: Structure-aware Shape Abstraction with Superquadrics for Generated Meshes 

**Title (ZH)**: Light-SQ: 基于超二次曲面的结构意识形状抽象生成网状结构 

**Authors**: Yuhan Wang, Weikai Chen, Zeyu Hu, Runze Zhang, Yingda Yin, Ruoyu Wu, Keyang Luo, Shengju Qian, Yiyan Ma, Hongyi Li, Yuan Gao, Yuhuan Zhou, Hao Luo, Wan Wang, Xiaobin Shen, Zhaowei Li, Kuixin Zhu, Chuanlang Hong, Yueyue Wang, Lijie Feng, Xin Wang, Chen Change Loy  

**Link**: [PDF](https://arxiv.org/pdf/2509.24986)  

**Abstract**: In user-generated-content (UGC) applications, non-expert users often rely on image-to-3D generative models to create 3D assets. In this context, primitive-based shape abstraction offers a promising solution for UGC scenarios by compressing high-resolution meshes into compact, editable representations. Towards this end, effective shape abstraction must therefore be structure-aware, characterized by low overlap between primitives, part-aware alignment, and primitive compactness. We present Light-SQ, a novel superquadric-based optimization framework that explicitly emphasizes structure-awareness from three aspects. (a) We introduce SDF carving to iteratively udpate the target signed distance field, discouraging overlap between primitives. (b) We propose a block-regrow-fill strategy guided by structure-aware volumetric decomposition, enabling structural partitioning to drive primitive placement. (c) We implement adaptive residual pruning based on SDF update history to surpress over-segmentation and ensure compact results. In addition, Light-SQ supports multiscale fitting, enabling localized refinement to preserve fine geometric details. To evaluate our method, we introduce 3DGen-Prim, a benchmark extending 3DGen-Bench with new metrics for both reconstruction quality and primitive-level editability. Extensive experiments demonstrate that Light-SQ enables efficient, high-fidelity, and editable shape abstraction with superquadrics for complex generated geometry, advancing the feasibility of 3D UGC creation. 

**Abstract (ZH)**: 基于超方程的结构感知3D形状抽象框架：Light-SQ 

---
# Segmentor-Guided Counterfactual Fine-Tuning for Image Synthesis 

**Title (ZH)**: 基于分割器引导的反事实微调图像合成 

**Authors**: Tian Xia, Matthew Sinclair, Andreas Schuh, Fabio De Sousa Ribeiro, Raghav Mehta, Rajat Rasal, Esther Puyol-Antón, Samuel Gerber, Kersten Petersen, Michiel Schaap, Ben Glocker  

**Link**: [PDF](https://arxiv.org/pdf/2509.24913)  

**Abstract**: Counterfactual image generation is a powerful tool for augmenting training data, de-biasing datasets, and modeling disease. Current approaches rely on external classifiers or regressors to increase the effectiveness of subject-level interventions (e.g., changing the patient's age). For structure-specific interventions (e.g., changing the area of the left lung in a chest radiograph), we show that this is insufficient, and can result in undesirable global effects across the image domain. Previous work used pixel-level label maps as guidance, requiring a user to provide hypothetical segmentations which are tedious and difficult to obtain. We propose Segmentor-guided Counterfactual Fine-Tuning (Seg-CFT), which preserves the simplicity of intervening on scalar-valued, structure-specific variables while producing locally coherent and effective counterfactuals. We demonstrate the capability of generating realistic chest radiographs, and we show promising results for modeling coronary artery disease. Code: this https URL. 

**Abstract (ZH)**: 基于分割图引导的反事实微调（Seg-CFT）：一种用于生成胸部X光片和模拟冠状动脉疾病的简便有效方法 

---
# OpenGPT-4o-Image: A Comprehensive Dataset for Advanced Image Generation and Editing 

**Title (ZH)**: OpenGPT-4o-Image：一个用于高级图像生成和编辑的综合数据集 

**Authors**: Zhihong Chen, Xuehai Bai, Yang Shi, Chaoyou Fu, Huanyu Zhang, Haotian Wang, Xiaoyan Sun, Zhang Zhang, Liang Wang, Yuanxing Zhang, Pengfei Wan, Yi-Fan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.24900)  

**Abstract**: The performance of unified multimodal models for image generation and editing is fundamentally constrained by the quality and comprehensiveness of their training data. While existing datasets have covered basic tasks like style transfer and simple object manipulation, they often lack the systematic structure and challenging scenarios required for real-world applications. To address this bottleneck, we introduce OpenGPT-4o-Image, a large-scale dataset constructed using a novel methodology that combines hierarchical task taxonomy with automated data generation. Our taxonomy not only includes fundamental capabilities such as text rendering and style control but also introduces highly practical yet challenging categories like scientific imagery for chemistry illustrations and complex instruction editing requiring simultaneous execution of multiple operations. Through an automated pipeline leveraging structured resource pools and GPT-4o, we generate 80k high-quality instruction-image pairs with controlled diversity, covering 11 major domains and 51 subtasks. Extensive experiments show that fine-tuning leading models on our dataset achieves significant performance gains across multiple benchmarks, with improvements of up to 18\% on editing tasks (UniWorld-V1 on ImgEdit-Bench) and 13% on generation tasks (Harmon on GenEval). Our work demonstrates that systematic data construction is key to advancing multimodal AI capabilities. 

**Abstract (ZH)**: 统一多模态模型在图像生成和编辑中的性能从根本上受到其训练数据质量和全面性的限制。现有数据集虽然覆盖了基本任务如风格迁移和简单的对象操作，但在系统结构和具有挑战性的场景方面仍不足以满足实际应用需求。为解决这一瓶颈，我们引入了OpenGPT-4o-Image，这是一种使用将层次任务分类学与自动化数据生成相结合的新方法构建的大规模数据集。我们的分类学不仅包括文本渲染和样式控制等基本能力，还引入了诸如化学插图所需的科学图像以及需要同时执行多个操作的复杂指令编辑等实用性极强但极具挑战性的类别。通过利用结构化资源池和GPT-4o的自动化管道，我们生成了80,000个高质量的指令-图像对，具有可控的多样性，覆盖了11个主要领域和51个子任务。广泛的实验表明，对我们的数据集进行微调，在多个基准测试中取得了显著性能提升，编辑任务的改进高达18%（UniWorld-V1在ImgEdit-Bench上），生成任务的改进高达13%（Harmon在GenEval上）。我们的工作表明，系统性数据构建是推动多模态AI能力进步的关键。 

---
# Causal-Adapter: Taming Text-to-Image Diffusion for Faithful Counterfactual Generation 

**Title (ZH)**: 因果适配器：驯化文本到图像扩散模型以实现符合事实的反事实生成 

**Authors**: Lei Tong, Zhihua Liu, Chaochao Lu, Dino Oglic, Tom Diethe, Philip Teare, Sotirios A. Tsaftaris, Chen Jin  

**Link**: [PDF](https://arxiv.org/pdf/2509.24798)  

**Abstract**: We present Causal-Adapter, a modular framework that adapts frozen text-to-image diffusion backbones for counterfactual image generation. Our method enables causal interventions on target attributes, consistently propagating their effects to causal dependents without altering the core identity of the image. In contrast to prior approaches that rely on prompt engineering without explicit causal structure, Causal-Adapter leverages structural causal modeling augmented with two attribute regularization strategies: prompt-aligned injection, which aligns causal attributes with textual embeddings for precise semantic control, and a conditioned token contrastive loss to disentangle attribute factors and reduce spurious correlations. Causal-Adapter achieves state-of-the-art performance on both synthetic and real-world datasets, with up to 91\% MAE reduction on Pendulum for accurate attribute control and 87\% FID reduction on ADNI for high-fidelity MRI image generation. These results show that our approach enables robust, generalizable counterfactual editing with faithful attribute modification and strong identity preservation. 

**Abstract (ZH)**: 我们提出Causal-Adapter，这是一种模块化框架，用于将冻结的文字到图像扩散骨干网络适应于生成反事实图像。该方法允许对目标属性进行因果干预，一致地传播其效应对因果依赖项，而不改变图像的核心身份。与依赖于提示工程而缺乏明确因果结构的先前方法不同，Causal-Adapter 利用增强的结构因果模型，并结合了两种属性正则化策略：提示对齐的注入，该策略使得因果属性与文本嵌入对齐以实现精确的语义控制，以及条件标记对比损失，以分离属性因素并减少虚假相关性。Causal-Adapter 在合成和真实世界数据集上均实现了最先进的性能，在Pendulum数据集上实现了高达91%的MAE减少，在ADNI数据集上实现了高达87%的FID减少，用于高保真MRI图像生成。这些结果表明，我们的方法能够实现稳健、具有泛化能力的反事实编辑，并且具有忠实的属性修改和强大的身份保留能力。 

---
# VSSFlow: Unifying Video-conditioned Sound and Speech Generation via Joint Learning 

**Title (ZH)**: VSSFlow：联合学习统一视频条件的声音和语音生成 

**Authors**: Xin Cheng, Yuyue Wang, Xihua Wang, Yihan Wu, Kaisi Guan, Yijing Chen, Peng Zhang, Xiaojiang Liu, Meng Cao, Ruihua Song  

**Link**: [PDF](https://arxiv.org/pdf/2509.24773)  

**Abstract**: Video-conditioned sound and speech generation, encompassing video-to-sound (V2S) and visual text-to-speech (VisualTTS) tasks, are conventionally addressed as separate tasks, with limited exploration to unify them within a signle framework. Recent attempts to unify V2S and VisualTTS face challenges in handling distinct condition types (e.g., heterogeneous video and transcript conditions) and require complex training stages. Unifying these two tasks remains an open problem. To bridge this gap, we present VSSFlow, which seamlessly integrates both V2S and VisualTTS tasks into a unified flow-matching framework. VSSFlow uses a novel condition aggregation mechanism to handle distinct input signals. We find that cross-attention and self-attention layer exhibit different inductive biases in the process of introducing condition. Therefore, VSSFlow leverages these inductive biases to effectively handle different representations: cross-attention for ambiguous video conditions and self-attention for more deterministic speech transcripts. Furthermore, contrary to the prevailing belief that joint training on the two tasks requires complex training strategies and may degrade performance, we find that VSSFlow benefits from the end-to-end joint learning process for sound and speech generation without extra designs on training stages. Detailed analysis attributes it to the learned general audio prior shared between tasks, which accelerates convergence, enhances conditional generation, and stabilizes the classifier-free guidance process. Extensive experiments demonstrate that VSSFlow surpasses the state-of-the-art domain-specific baselines on both V2S and VisualTTS benchmarks, underscoring the critical potential of unified generative models. 

**Abstract (ZH)**: 视频条件下的声音和语音生成：VSSFlow统一视频到声音和视觉文本到语音任务 

---
# SANA-Video: Efficient Video Generation with Block Linear Diffusion Transformer 

**Title (ZH)**: SANA-视频：块线性扩散变换器驱动的高效视频生成 

**Authors**: Junsong Chen, Yuyang Zhao, Jincheng Yu, Ruihang Chu, Junyu Chen, Shuai Yang, Xianbang Wang, Yicheng Pan, Daquan Zhou, Huan Ling, Haozhe Liu, Hongwei Yi, Hao Zhang, Muyang Li, Yukang Chen, Han Cai, Sanja Fidler, Ping Luo, Song Han, Enze Xie  

**Link**: [PDF](https://arxiv.org/pdf/2509.24695)  

**Abstract**: We introduce SANA-Video, a small diffusion model that can efficiently generate videos up to 720x1280 resolution and minute-length duration. SANA-Video synthesizes high-resolution, high-quality and long videos with strong text-video alignment at a remarkably fast speed, deployable on RTX 5090 GPU. Two core designs ensure our efficient, effective and long video generation: (1) Linear DiT: We leverage linear attention as the core operation, which is more efficient than vanilla attention given the large number of tokens processed in video generation. (2) Constant-Memory KV cache for Block Linear Attention: we design block-wise autoregressive approach for long video generation by employing a constant-memory state, derived from the cumulative properties of linear attention. This KV cache provides the Linear DiT with global context at a fixed memory cost, eliminating the need for a traditional KV cache and enabling efficient, minute-long video generation. In addition, we explore effective data filters and model training strategies, narrowing the training cost to 12 days on 64 H100 GPUs, which is only 1% of the cost of MovieGen. Given its low cost, SANA-Video achieves competitive performance compared to modern state-of-the-art small diffusion models (e.g., Wan 2.1-1.3B and SkyReel-V2-1.3B) while being 16x faster in measured latency. Moreover, SANA-Video can be deployed on RTX 5090 GPUs with NVFP4 precision, accelerating the inference speed of generating a 5-second 720p video from 71s to 29s (2.4x speedup). In summary, SANA-Video enables low-cost, high-quality video generation. 

**Abstract (ZH)**: SANA-Video：一种高效生成高清长视频的小规模扩散模型 

---
# VNODE: A Piecewise Continuous Volterra Neural Network 

**Title (ZH)**: VNODE：分段连续维特拉神经网络 

**Authors**: Siddharth Roheda, Aniruddha Bala, Rohit Chowdhury, Rohan Jaiswal  

**Link**: [PDF](https://arxiv.org/pdf/2509.24659)  

**Abstract**: This paper introduces Volterra Neural Ordinary Differential Equations (VNODE), a piecewise continuous Volterra Neural Network that integrates nonlinear Volterra filtering with continuous time neural ordinary differential equations for image classification. Drawing inspiration from the visual cortex, where discrete event processing is interleaved with continuous integration, VNODE alternates between discrete Volterra feature extraction and ODE driven state evolution. This hybrid formulation captures complex patterns while requiring substantially fewer parameters than conventional deep architectures. VNODE consistently outperforms state of the art models with improved computational complexity as exemplified on benchmark datasets like CIFAR10 and Imagenet1K. 

**Abstract (ZH)**: Volterra神经常微分方程（VNODE）：一种用于图像分类的分段连续Volterra神经网络 

---
# Can you SPLICE it together? A Human Curated Benchmark for Probing Visual Reasoning in VLMs 

**Title (ZH)**: 你能将它们拼接起来吗？一种人工策展的视觉推理基准测试用于探查多模态视觉语言模型 

**Authors**: Mohamad Ballout, Okajevo Wilfred, Seyedalireza Yaghoubi, Nohayr Muhammad Abdelmoneim, Julius Mayer, Elia Bruni  

**Link**: [PDF](https://arxiv.org/pdf/2509.24640)  

**Abstract**: In this work, we introduce SPLICE, a human-curated benchmark derived from the COIN instructional video dataset, designed to probe event-based reasoning across multiple dimensions: temporal, causal, spatial, contextual, and general knowledge. SPLICE includes 3,381 human-filtered videos spanning 12 categories and 180 sub-categories, such as sports, engineering, and housework. These videos are segmented into a total of 11,423 event clips. We evaluate both human participants and state-of-the-art vision-language models (VLMs) on the task of rearranging these clips into coherent event sequences to assess visual reasoning capabilities. Results reveal a significant gap: VLMs struggle to match human performance. While human-annotated textual descriptions improve model accuracy, they do not affect human performance, suggesting that models rely more on language priors than on visual understanding. Even with annotations, VLMs fall short of human-level reasoning, underscoring persistent challenges in visual reasoning. A deeper analysis across sub-categories shows that VLMs perform relatively better on videos where temporal and causal reasoning are dominant, compared to those where contextual and spatial reasoning are dominant. They also perform better on everyday tasks than on specialized ones. 

**Abstract (ZH)**: 本研究引入了SPLICE，一个由人类整理的基准数据集，源自COIN指令视频数据集，旨在从时间、因果关系、空间、上下文及通用知识多个维度探讨事件推理。SPLICE包括3,381个人筛选过的视频，涵盖12个类别和180个子类别，如体育、工程和家务。这些视频被分割成共计11,423个事件片段。我们评估了人类参与者和最先进的视觉-语言模型（VLMs）重新排列这些片段以形成连贯事件序列的能力，以评估其视觉推理能力。结果显示，视觉模型与人类表现存在显著差距：尽管人类注释的文本描述可以提高模型的准确性，但不足以影响人类的表现，表明模型更多依赖于语言先验而非视觉理解。即使有注释，视觉模型仍无法达到人类级别的推理水平，突显了视觉推理领域的持续挑战。通过对子类别的深入分析发现，在以时间和因果关系推理为主导的视频上，视觉模型表现优于以上下文和空间推理为主导的视频；在日常任务上表现优于专门任务。 

---
# LaMoGen: Laban Movement-Guided Diffusion for Text-to-Motion Generation 

**Title (ZH)**: Laban 动作引导的文本到动作生成扩散模型 

**Authors**: Heechang Kim, Gwanghyun Kim, Se Young Chun  

**Link**: [PDF](https://arxiv.org/pdf/2509.24469)  

**Abstract**: Diverse human motion generation is an increasingly important task, having various applications in computer vision, human-computer interaction and animation. While text-to-motion synthesis using diffusion models has shown success in generating high-quality motions, achieving fine-grained expressive motion control remains a significant challenge. This is due to the lack of motion style diversity in datasets and the difficulty of expressing quantitative characteristics in natural language. Laban movement analysis has been widely used by dance experts to express the details of motion including motion quality as consistent as possible. Inspired by that, this work aims for interpretable and expressive control of human motion generation by seamlessly integrating the quantification methods of Laban Effort and Shape components into the text-guided motion generation models. Our proposed zero-shot, inference-time optimization method guides the motion generation model to have desired Laban Effort and Shape components without any additional motion data by updating the text embedding of pretrained diffusion models during the sampling step. We demonstrate that our approach yields diverse expressive motion qualities while preserving motion identity by successfully manipulating motion attributes according to target Laban tags. 

**Abstract (ZH)**: 多样化的探究性人类运动生成是一项日益重要的任务，广泛应用于计算机视觉、人机交互和动画领域。虽然利用扩散模型进行文本到运动的合成已成功生成高质量的运动，但实现精细的表达性运动控制仍然是一个重大挑战。这源于数据集中运动风格多样性不足以及通过自然语言表达定量特征的困难。朗班运动分析已被舞蹈专家广泛用于表达运动细节，包括尽量一致的运动质量。受此启发，本工作旨在通过无缝整合朗班力量和形态成分的量化方法，实现由文本指导的人类运动生成的可解释和表达性控制。我们提出了一种零-shot，在推断时优化的方法，通过在采样步骤中更新预训练扩散模型的文本嵌入，指导运动生成模型生成所需的目标朗班力量和形态成分，而无需额外的运动数据。我们证明，通过根据目标朗班标签操控运动属性，我们的方法能够在保持运动身份的同时产生多样化的表达性运动质量。 

---
# A Data-Centric Perspective on the Influence of Image Data Quality in Machine Learning Models 

**Title (ZH)**: 基于数据为中心的观点：图像数据质量对机器学习模型的影响研究 

**Authors**: Pei-Han Chen, Szu-Chi Chung  

**Link**: [PDF](https://arxiv.org/pdf/2509.24420)  

**Abstract**: In machine learning, research has traditionally focused on model development, with relatively less attention paid to training data. As model architectures have matured and marginal gains from further refinements diminish, data quality has emerged as a critical factor. However, systematic studies on evaluating and ensuring dataset quality in the image domain remain limited.
This study investigates methods for systematically assessing image dataset quality and examines how various image quality factors influence model performance. Using the publicly available and relatively clean CIFAKE dataset, we identify common quality issues and quantify their impact on training. Building on these findings, we develop a pipeline that integrates two community-developed tools, CleanVision and Fastdup. We analyze their underlying mechanisms and introduce several enhancements, including automatic threshold selection to detect problematic images without manual tuning.
Experimental results demonstrate that not all quality issues exert the same level of impact. While convolutional neural networks show resilience to certain distortions, they are particularly vulnerable to degradations that obscure critical visual features, such as blurring and severe downscaling. To assess the performance of existing tools and the effectiveness of our proposed enhancements, we formulate the detection of low-quality images as a binary classification task and use the F1 score as the evaluation metric. Our automatic thresholding method improves the F1 score from 0.6794 to 0.9468 under single perturbations and from 0.7447 to 0.8557 under dual perturbations. For near-duplicate detection, our deduplication strategy increases the F1 score from 0.4576 to 0.7928. These results underscore the effectiveness of our workflow and provide a foundation for advancing data quality assessment in image-based machine learning. 

**Abstract (ZH)**: 基于图像的数据集质量系统评估方法及其对模型性能的影响 

---
# REALIGN: Regularized Procedure Alignment with Matching Video Embeddings via Partial Gromov-Wasserstein Optimal Transport 

**Title (ZH)**: REALIGN: 正则化程序对齐通过部分Gromov-Wasserstein最优传输匹配视频嵌入 

**Authors**: Soumyadeep Chandra, Kaushik Roy  

**Link**: [PDF](https://arxiv.org/pdf/2509.24382)  

**Abstract**: Learning from procedural videos remains a core challenge in self-supervised representation learning, as real-world instructional data often contains background segments, repeated actions, and steps presented out of order. Such variability violates the strong monotonicity assumptions underlying many alignment methods. Prior state-of-the-art approaches, such as OPEL, leverage Kantorovich Optimal Transport (KOT) to build frame-to-frame correspondences, but rely solely on feature similarity and fail to capture the higher-order temporal structure of a task. In this paper, we introduce REALIGN, a self-supervised framework for procedure learning based on Regularized Fused Partial Gromov-Wasserstein Optimal Transport (R-FPGWOT). In contrast to KOT, our formulation jointly models visual correspondences and temporal relations under a partial alignment scheme, enabling robust handling of irrelevant frames, repeated actions, and non-monotonic step orders common in instructional videos. To stabilize training, we integrate FPGWOT distances with inter-sequence contrastive learning, avoiding the need for multiple regularizers and preventing collapse to degenerate solutions. Across egocentric (EgoProceL) and third-person (ProceL, CrossTask) benchmarks, REALIGN achieves up to 18.9% average F1-score improvements and over 30% temporal IoU gains, while producing more interpretable transport maps that preserve key-step orderings and filter out noise. 

**Abstract (ZH)**: 基于正则化融合部分Gromov-Wasserstein最优传输的自监督程序学习 

---
# From Satellite to Street: A Hybrid Framework Integrating Stable Diffusion and PanoGAN for Consistent Cross-View Synthesis 

**Title (ZH)**: 从卫星到街道：一种结合 Stable Diffusion 和 PanoGAN 的混合框架，用于一致的跨视角合成 

**Authors**: Khawlah Bajbaa, Abbas Anwar, Muhammad Saqib, Hafeez Anwar, Nabin Sharma, Muhammad Usman  

**Link**: [PDF](https://arxiv.org/pdf/2509.24369)  

**Abstract**: Street view imagery has become an essential source for geospatial data collection and urban analytics, enabling the extraction of valuable insights that support informed decision-making. However, synthesizing street-view images from corresponding satellite imagery presents significant challenges due to substantial differences in appearance and viewing perspective between these two domains. This paper presents a hybrid framework that integrates diffusion-based models and conditional generative adversarial networks to generate geographically consistent street-view images from satellite imagery. Our approach uses a multi-stage training strategy that incorporates Stable Diffusion as the core component within a dual-branch architecture. To enhance the framework's capabilities, we integrate a conditional Generative Adversarial Network (GAN) that enables the generation of geographically consistent panoramic street views. Furthermore, we implement a fusion strategy that leverages the strengths of both models to create robust representations, thereby improving the geometric consistency and visual quality of the generated street-view images. The proposed framework is evaluated on the challenging Cross-View USA (CVUSA) dataset, a standard benchmark for cross-view image synthesis. Experimental results demonstrate that our hybrid approach outperforms diffusion-only methods across multiple evaluation metrics and achieves competitive performance compared to state-of-the-art GAN-based methods. The framework successfully generates realistic and geometrically consistent street-view images while preserving fine-grained local details, including street markings, secondary roads, and atmospheric elements such as clouds. 

**Abstract (ZH)**: 街景视图图像已成为地理空间数据收集和城市分析的重要来源，能够提取有价值的信息以支持明智的决策。然而，从对应的卫星图像合成街景图像由于这两个领域在外观和视角上的显著差异而面临重大挑战。本文提出了一种结合扩散模型和条件生成对抗网络的混合框架，以从卫星图像生成地理一致的街景图像。我们的方法采用多阶段训练策略，并将Stable Diffusion作为核心组件集成到双重分支架构中。为了增强框架的能力，我们引入了一个条件生成对抗网络（GAN），以生成地理一致的全景街景。此外，我们实施了一种融合策略，利用两者的优点，从而提高生成街景图像的几何一致性和视觉质量。所提出框架在Cross-View USA (CVUSA) 数据集上进行了评估，该数据集是交叉视图图像合成的标准化基准。实验结果表明，我们的混合方法在多个评估指标上优于仅扩散的方法，并且在与最先进的GAN基方法的性能上具有竞争力。该框架成功生成了具有真实感和几何一致性的街景图像，同时保留了详细的局部细节，包括街道标记、次要道路和大气元素如云彩。 

---
# An Enhanced Pyramid Feature Network Based on Long-Range Dependencies for Multi-Organ Medical Image Segmentation 

**Title (ZH)**: 基于长程依赖关系的增强 pyramid 特征网络多器官医疗图像分割 

**Authors**: Dayu Tan, Cheng Kong, Yansen Su, Hai Chen, Dongliang Yang, Junfeng Xia, Chunhou Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.24358)  

**Abstract**: In the field of multi-organ medical image segmentation, recent methods frequently employ Transformers to capture long-range dependencies from image features. However, these methods overlook the high computational cost of Transformers and their deficiencies in extracting local detailed information. To address high computational costs and inadequate local detail information, we reassess the design of feature extraction modules and propose a new deep-learning network called LamFormer for fine-grained segmentation tasks across multiple organs. LamFormer is a novel U-shaped network that employs Linear Attention Mamba (LAM) in an enhanced pyramid encoder to capture multi-scale long-range dependencies. We construct the Parallel Hierarchical Feature Aggregation (PHFA) module to aggregate features from different layers of the encoder, narrowing the semantic gap among features while filtering information. Finally, we design the Reduced Transformer (RT), which utilizes a distinct computational approach to globally model up-sampled features. RRT enhances the extraction of detailed local information and improves the network's capability to capture long-range dependencies. LamFormer outperforms existing segmentation methods on seven complex and diverse datasets, demonstrating exceptional performance. Moreover, the proposed network achieves a balance between model performance and model complexity. 

**Abstract (ZH)**: 多器官医学图像分割领域的LamFormer：平衡高性能与低复杂度的新型深度学习网络 

---
# Dynamic Orchestration of Multi-Agent System for Real-World Multi-Image Agricultural VQA 

**Title (ZH)**: 面向现实多图像农业VQA的多agents系统动态 orchestration 

**Authors**: Yan Ke, Xin Yu, Heming Du, Scott Chapman, Helen Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.24350)  

**Abstract**: Agricultural visual question answering is essential for providing farmers and researchers with accurate and timely knowledge. However, many existing approaches are predominantly developed for evidence-constrained settings such as text-only queries or single-image cases. This design prevents them from coping with real-world agricultural scenarios that often require multi-image inputs with complementary views across spatial scales, and growth stages. Moreover, limited access to up-to-date external agricultural context makes these systems struggle to adapt when evidence is incomplete. In addition, rigid pipelines often lack systematic quality control. To address this gap, we propose a self-reflective and self-improving multi-agent framework that integrates four roles, the Retriever, the Reflector, the Answerer, and the Improver. They collaborate to enable context enrichment, reflective reasoning, answer drafting, and iterative improvement.
A Retriever formulates queries and gathers external information, while a Reflector assesses adequacy and triggers sequential reformulation and renewed retrieval. Two Answerers draft candidate responses in parallel to reduce bias. The Improver refines them through iterative checks while ensuring that information from multiple images is effectively aligned and utilized. Experiments on the AgMMU benchmark show that our framework achieves competitive performance on multi-image agricultural QA. 

**Abstract (ZH)**: 农业视觉问答对于为农民和研究人员提供准确及时的知识至关重要。然而，许多现有方法主要针对如仅文本查询或单张图像等证据受限的场景进行开发。这种设计使其难以应对需要跨空间尺度和生长阶段互补视角的多张图像输入的真实农业场景，并且在证据不完整时难以适应。此外，固定的流程往往缺乏系统性的质量控制。为解决这一问题，我们提出了一种自我反思和自我改进的多agent框架，整合了检索者、反思者、回答者和改进者四个角色。它们协作以实现上下文丰富、反思推理、回答草拟及迭代改进。检索者制定问题并收集外部信息，反思者评估其充分性并触发顺序重新表述和重新检索。两个回答者并行草拟候选回答以减少偏见。改进者通过迭代检查改善回答，同时确保来自多张图像的信息得到有效协调和利用。在AgMMU基准测试上的实验表明，我们的框架在多图农业问答任务中取得了竞争力的表现。 

---
# Cycle Diffusion Model for Counterfactual Image Generation 

**Title (ZH)**: 循环扩散模型用于反事实图像生成 

**Authors**: Fangrui Huang, Alan Wang, Binxu Li, Bailey Trang, Ridvan Yesiloglu, Tianyu Hua, Wei Peng, Ehsan Adeli  

**Link**: [PDF](https://arxiv.org/pdf/2509.24267)  

**Abstract**: Deep generative models have demonstrated remarkable success in medical image synthesis. However, ensuring conditioning faithfulness and high-quality synthetic images for direct or counterfactual generation remains a challenge. In this work, we introduce a cycle training framework to fine-tune diffusion models for improved conditioning adherence and enhanced synthetic image realism. Our approach, Cycle Diffusion Model (CDM), enforces consistency between generated and original images by incorporating cycle constraints, enabling more reliable direct and counterfactual generation. Experiments on a combined 3D brain MRI dataset (from ABCD, HCP aging & young adults, ADNI, and PPMI) show that our method improves conditioning accuracy and enhances image quality as measured by FID and SSIM. The results suggest that the cycle strategy used in CDM can be an effective method for refining diffusion-based medical image generation, with applications in data augmentation, counterfactual, and disease progression modeling. 

**Abstract (ZH)**: 深生成模型在医学图像合成中取得了显著成功，但确保条件忠实性和高质量的合成图像以进行直接生成或反事实生成仍是挑战。本文介绍了一种循环训练框架，以微调扩散模型，从而提高条件依从性和增强合成图像的真实感。我们的方法，循环扩散模型（CDM），通过引入循环约束确保生成图像与原始图像的一致性，从而实现更可靠的直接生成和反事实生成。在合并的3D脑MRI数据集（来自ABCD、HCP老化与年轻成人、ADNI和PPMI）上的实验表明，我们的方法可以提高条件准确性和通过FID和SSIM衡量的图像质量。结果表明，CDM中使用的循环策略可以有效改进基于扩散的医学图像生成，适用于数据增强、反事实和疾病进展建模。 

---
# BALR-SAM: Boundary-Aware Low-Rank Adaptation of SAM for Resource-Efficient Medical Image Segmentation 

**Title (ZH)**: 边界感知低秩适应SAM资源高效医疗图像分割 

**Authors**: Zelin Liu, Sicheng Dong, Bocheng Li, Yixuan Yang, Jiacheng Ruan, Chenxu Zhou, Suncheng Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2509.24204)  

**Abstract**: Vision foundation models like the Segment Anything Model (SAM), pretrained on large-scale natural image datasets, often struggle in medical image segmentation due to a lack of domain-specific adaptation. In clinical practice, fine-tuning such models efficiently for medical downstream tasks with minimal resource demands, while maintaining strong performance, is challenging. To address these issues, we propose BALR-SAM, a boundary-aware low-rank adaptation framework that enhances SAM for medical imaging. It combines three tailored components: (1) a Complementary Detail Enhancement Network (CDEN) using depthwise separable convolutions and multi-scale fusion to capture boundary-sensitive features essential for accurate segmentation; (2) low-rank adapters integrated into SAM's Vision Transformer blocks to optimize feature representation and attention for medical contexts, while simultaneously significantly reducing the parameter space; and (3) a low-rank tensor attention mechanism in the mask decoder, cutting memory usage by 75% and boosting inference speed. Experiments on standard medical segmentation datasets show that BALR-SAM, without requiring prompts, outperforms several state-of-the-art (SOTA) methods, including fully fine-tuned MedSAM, while updating just 1.8% (11.7M) of its parameters. 

**Abstract (ZH)**: 边界感知低秩适配框架BALR-SAM：一种用于医学影像分割的Segment Anything Model增强方法 

---
# Talk in Pieces, See in Whole: Disentangling and Hierarchical Aggregating Representations for Language-based Object Detection 

**Title (ZH)**: 破碎片段之谈，观整体之象：基于语言的对象检测中表示的解耦与分层聚合 

**Authors**: Sojung An, Kwanyong Park, Yong Jae Lee, Donghyun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2509.24192)  

**Abstract**: While vision-language models (VLMs) have made significant progress in multimodal perception (e.g., open-vocabulary object detection) with simple language queries, state-of-the-art VLMs still show limited ability to perceive complex queries involving descriptive attributes and relational clauses. Our in-depth analysis shows that these limitations mainly stem from text encoders in VLMs. Such text encoders behave like bags-of-words and fail to separate target objects from their descriptive attributes and relations in complex queries, resulting in frequent false positives. To address this, we propose restructuring linguistic representations according to the hierarchical relations within sentences for language-based object detection. A key insight is the necessity of disentangling textual tokens into core components-objects, attributes, and relations ("talk in pieces")-and subsequently aggregating them into hierarchically structured sentence-level representations ("see in whole"). Building on this principle, we introduce the TaSe framework with three main contributions: (1) a hierarchical synthetic captioning dataset spanning three tiers from category names to descriptive sentences; (2) Talk in Pieces, the three-component disentanglement module guided by a novel disentanglement loss function, transforms text embeddings into subspace compositions; and (3) See in Whole, which learns to aggregate disentangled components into hierarchically structured embeddings with the guide of proposed hierarchical objectives. The proposed TaSe framework strengthens the inductive bias of hierarchical linguistic structures, resulting in fine-grained multimodal representations for language-based object detection. Experimental results under the OmniLabel benchmark show a 24% performance improvement, demonstrating the importance of linguistic compositionality. 

**Abstract (ZH)**: 基于语言的物体检测中视觉-语言模型的层次化语言表示重构 

---
# Accelerating Cerebral Diagnostics with BrainFusion: A Comprehensive MRI Tumor Framework 

**Title (ZH)**: 基于BrainFusion的全面MRI肿瘤框架加速脑部诊断 

**Authors**: Walid Houmaidi, Youssef Sabiri, Salmane El Mansour Billah, Amine Abouaomar  

**Link**: [PDF](https://arxiv.org/pdf/2509.24149)  

**Abstract**: The early and accurate classification of brain tumors is crucial for guiding effective treatment strategies and improving patient outcomes. This study presents BrainFusion, a significant advancement in brain tumor analysis using magnetic resonance imaging (MRI) by combining fine-tuned convolutional neural networks (CNNs) for tumor classification--including VGG16, ResNet50, and Xception--with YOLOv8 for precise tumor localization with bounding boxes. Leveraging the Brain Tumor MRI Dataset, our experiments reveal that the fine-tuned VGG16 model achieves test accuracy of 99.86%, substantially exceeding previous benchmarks. Beyond setting a new accuracy standard, the integration of bounding-box localization and explainable AI techniques further enhances both the clinical interpretability and trustworthiness of the system's outputs. Overall, this approach underscores the transformative potential of deep learning in delivering faster, more reliable diagnoses, ultimately contributing to improved patient care and survival rates. 

**Abstract (ZH)**: 脑肿瘤的早期和准确分类对于指导有效的治疗策略和改善患者预后至关重要。本研究提出BrainFusion，这是一种通过结合微调的卷积神经网络（包括VGG16、ResNet50和Xception）与YOLOv8进行精准肿瘤定位的磁共振成像（MRI）脑肿瘤分析的显著进展。利用Brain Tumor MRI数据集，我们的实验表明微调的VGG16模型在测试集上的准确率为99.86%，显著超过先前的标准。此外，结合边界框定位和可解释AI技术进一步增强了系统输出的临床可解释性和可靠性。总体而言，该方法强调了深度学习在实现更快、更可靠诊断方面的变革潜力，最终有助于改善患者的护理质量和生存率。 

---
# EYE-DEX: Eye Disease Detection and EXplanation System 

**Title (ZH)**: EYE-DEX：眼科疾病检测与解释系统 

**Authors**: Youssef Sabiri, Walid Houmaidi, Amine Abouaomar  

**Link**: [PDF](https://arxiv.org/pdf/2509.24136)  

**Abstract**: Retinal disease diagnosis is critical in preventing vision loss and reducing socioeconomic burdens. Globally, over 2.2 billion people are affected by some form of vision impairment, resulting in annual productivity losses estimated at $411 billion. Traditional manual grading of retinal fundus images by ophthalmologists is time-consuming and subjective. In contrast, deep learning has revolutionized medical diagnostics by automating retinal image analysis and achieving expert-level performance. In this study, we present EYE-DEX, an automated framework for classifying 10 retinal conditions using the large-scale Retinal Disease Dataset comprising 21,577 eye fundus images. We benchmark three pre-trained Convolutional Neural Network (CNN) models--VGG16, VGG19, and ResNet50--with our finetuned VGG16 achieving a state-of-the-art global benchmark test accuracy of 92.36%. To enhance transparency and explainability, we integrate the Gradient-weighted Class Activation Mapping (Grad-CAM) technique to generate visual explanations highlighting disease-specific regions, thereby fostering clinician trust and reliability in AI-assisted diagnostics. 

**Abstract (ZH)**: 视网膜疾病诊断对于预防视力丧失和减轻社会经济负担至关重要。全球有超过22亿人受到不同程度的视力障碍影响，导致年度生产力损失估计达到4110亿美元。传统的眼科医生 manually 对视网膜底片图像进行分级耗时且主观。相比之下，深度学习已经通过自动化视网膜图像分析实现了专家级性能，从而革命了医疗诊断。在本研究中，我们提出了一种名为EYE-DEX的自动化框架，用于分类10种视网膜疾病，该框架基于包含21,577张眼底图像的大型视网膜疾病数据集。我们基准测试了三种预训练的卷积神经网络模型——VGG16、VGG19和ResNet50，其中我们细调的VGG16模型在全局基准测试中的准确率达到92.36%，处于领先地位。为了提高透明度和可解释性，我们整合了梯度加权类激活映射（Grad-CAM）技术，生成视觉解释，突出显示疾病特异性区域，从而增强临床医生对人工智能辅助诊断的信任和可靠性。 

---
# FrameMind: Frame-Interleaved Chain-of-Thought for Video Reasoning via Reinforcement Learning 

**Title (ZH)**: FrameMind: 帧间插/frame交替链式思考的视频推理方法 via 强化学习 

**Authors**: Haonan Ge, Yiwei Wang, Kai-Wei Chang, Hang Wu, Yujun Cai  

**Link**: [PDF](https://arxiv.org/pdf/2509.24008)  

**Abstract**: Current video understanding models rely on fixed frame sampling strategies, processing predetermined visual inputs regardless of the specific reasoning requirements of each question. This static approach limits their ability to adaptively gather visual evidence, leading to suboptimal performance on tasks that require either broad temporal coverage or fine-grained spatial detail. In this paper, we introduce FrameMind, an end-to-end framework trained with reinforcement learning that enables models to dynamically request visual information during reasoning through Frame-Interleaved Chain-of-Thought (FiCOT). Unlike traditional approaches, FrameMind operates in multiple turns where the model alternates between textual reasoning and active visual perception, using tools to extract targeted frames or video clips based on identified knowledge gaps. To train effective dynamic sampling policies, we propose Dynamic Resolution Frame Sampling (DRFS), which exposes models to diverse temporal-spatial trade-offs during learning, and DRFS-GRPO, a group-relative policy optimization algorithm that learns from outcome-based rewards without requiring frame-level annotations. Extensive experiments on challenging benchmarks like MLVU and VideoMME demonstrate that our method significantly outperforms existing models, advancing the state of the art in flexible and efficient video understanding. 

**Abstract (ZH)**: 当前的视频理解模型依赖于固定帧采样策略，在推理过程中处理预先确定的视觉输入，而不考虑每个问题的具体推理需求。这种静态方法限制了模型适应性地收集视觉证据的能力，导致在需要广泛的时间覆盖或精细的空间细节的任务中表现不佳。本文引入了FrameMind，这是一种通过帧插混思维链（FiCOT）训练的端到端框架，使模型能够在推理过程中动态请求视觉信息。FrameMind在多轮交互中运作，模型交替进行文本推理和主动视觉感知，使用工具根据识别的知识缺口提取目标帧或视频片段。为了训练有效的动态采样策略，我们提出了动态分辨率帧采样（DRFS），让模型在学习过程中暴露于多样化的时空间权衡中，并使用基于结果的奖励学习群相对策优化算法（DRFS-GRPO），无需帧级注释。在如MLVU和VideoMME等具有挑战性的基准测试中，我们的方法显著优于现有模型，推动了灵活高效视频理解的技术进步。 

---
# SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse-Linear Attention 

**Title (ZH)**: SLA：通过可微调稀疏线性注意力超越扩散变换器的稀疏性 

**Authors**: Jintao Zhang, Haoxu Wang, Kai Jiang, Shuo Yang, Kaiwen Zheng, Haocheng Xi, Ziteng Wang, Hongzhou Zhu, Min Zhao, Ion Stoica, Joseph E. Gonzalez, Jun Zhu, Jianfei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.24006)  

**Abstract**: In Diffusion Transformer (DiT) models, particularly for video generation, attention latency is a major bottleneck due to the long sequence length and the quadratic complexity. We find that attention weights can be separated into two parts: a small fraction of large weights with high rank and the remaining weights with very low rank. This naturally suggests applying sparse acceleration to the first part and low-rank acceleration to the second. Based on this finding, we propose SLA (Sparse-Linear Attention), a trainable attention method that fuses sparse and linear attention to accelerate diffusion models. SLA classifies attention weights into critical, marginal, and negligible categories, applying O(N^2) attention to critical weights, O(N) attention to marginal weights, and skipping negligible ones. SLA combines these computations into a single GPU kernel and supports both forward and backward passes. With only a few fine-tuning steps using SLA, DiT models achieve a 20x reduction in attention computation, resulting in significant acceleration without loss of generation quality. Experiments show that SLA reduces attention computation by 95% without degrading end-to-end generation quality, outperforming baseline methods. In addition, we implement an efficient GPU kernel for SLA, which yields a 13.7x speedup in attention computation and a 2.2x end-to-end speedup in video generation on Wan2.1-1.3B. 

**Abstract (ZH)**: 在Diffusion Transformer（DiT）模型中，尤其是对于视频生成，注意力延迟由于长序列长度和 quadratic 复杂性成为主要瓶颈。我们发现注意力权重可以分为两部分：一小部分具有高秩的大权重和剩余部分具有极低秩的权重。这自然地提示我们对前一部分使用稀疏加速，对后一部分使用低秩加速。基于这一发现，我们提出了SLA（Sparse-Linear Attention），一种可训练的注意力方法，将稀疏注意力和线性注意力融合以加速扩散模型。SLA将注意力权重分为关键、边缘和可忽略不计三类，对关键权重应用 O(N^2) 注意力，对边缘权重应用 O(N) 注意力，并跳过可忽略不计的权重。SLA将这些计算合并到一个 GPU 核심，并支持前向和反向传播。仅通过少量的 SLA 微调步骤，DiT 模型实现了注意力计算20倍的减少，显著加速同时不损失生成质量。实验表明，SLA 在不降低端到端生成质量的情况下，将注意力计算减少95%，并优于基线方法。此外，我们为 SLA 实现了一个高效的 GPU 核心，使其在 Wan2.1-1.3B 上的注意力计算加速13.7倍，在视频生成中端到端加速2.2倍。 

---
# Disentangling Score Content and Performance Style for Joint Piano Rendering and Transcription 

**Title (ZH)**: 分离评分内容和表演风格以实现联合钢琴渲染与转记 

**Authors**: Wei Zeng, Junchuan Zhao, Ye Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23878)  

**Abstract**: Expressive performance rendering (EPR) and automatic piano transcription (APT) are fundamental yet inverse tasks in music information retrieval: EPR generates expressive performances from symbolic scores, while APT recovers scores from performances. Despite their dual nature, prior work has addressed them independently. In this paper we propose a unified framework that jointly models EPR and APT by disentangling note-level score content and global performance style representations from both paired and unpaired data. Our framework is built on a transformer-based sequence-to-sequence architecture and is trained using only sequence-aligned data, without requiring fine-grained note-level alignment. To automate the rendering process while ensuring stylistic compatibility with the score, we introduce an independent diffusion-based performance style recommendation module that generates style embeddings directly from score content. This modular component supports both style transfer and flexible rendering across a range of expressive styles. Experimental results from both objective and subjective evaluations demonstrate that our framework achieves competitive performance on EPR and APT tasks, while enabling effective content-style disentanglement, reliable style transfer, and stylistically appropriate rendering. Demos are available at this https URL 

**Abstract (ZH)**: 表达性表演渲染和自动钢琴转录的统一框架：从符号乐谱中生成表达性表演与从表演中恢复乐谱是音乐信息检索中的基本且相互逆向的任务：表达性表演渲染生成表达性表演，而自动钢琴转录则从表演恢复乐谱。尽管它们是相互逆向的任务，但以往的工作却分别处理它们。本文提出一个统一框架，通过从配对和未配对数据中分离出音级乐谱内容和全局表演风格表示，联合建模表达性表演渲染和自动钢琴转录。该框架基于基于变压器的序列到序列架构，并仅使用序列对齐的数据进行训练，无需进行细粒度的音级对齐。为了自动化渲染过程并确保与乐谱风格的一致性，我们引入了一个独立的基于扩散的表演风格推荐模块，该模块可以直接从乐谱内容生成风格嵌入。该模块支持风格转移和各种表达性风格的灵活渲染。客观和主观评估实验结果表明，该框架在表达性表演渲染和自动钢琴转录任务上达到了竞争力，并且能够有效地分离内容和风格，可靠地进行风格转移，并生成风格适当的渲染。演示可在以下链接获取：this https URL 

---
# Not All Tokens are Guided Equal: Improving Guidance in Visual Autoregressive Models 

**Title (ZH)**: 不同指导token并非平等：提高视觉自回归模型中的指导效果 

**Authors**: Ky Dan Nguyen, Hoang Lam Tran, Anh-Dung Dinh, Daochang Liu, Weidong Cai, Xiuying Wang, Chang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.23876)  

**Abstract**: Autoregressive (AR) models based on next-scale prediction are rapidly emerging as a powerful tool for image generation, but they face a critical weakness: information inconsistencies between patches across timesteps introduced by progressive resolution scaling. These inconsistencies scatter guidance signals, causing them to drift away from conditioning information and leaving behind ambiguous, unfaithful features. We tackle this challenge with Information-Grounding Guidance (IGG), a novel mechanism that anchors guidance to semantically important regions through attention. By adaptively reinforcing informative patches during sampling, IGG ensures that guidance and content remain tightly aligned. Across both class-conditioned and text-to-image generation tasks, IGG delivers sharper, more coherent, and semantically grounded images, setting a new benchmark for AR-based methods. 

**Abstract (ZH)**: 基于下一级预测的自回归（AR）模型正在迅速成为图像生成的一个强大工具，但它们面临一个关键弱点：逐级分辨率缩放导致的时间步长内 patch 间的信息不一致性。这些不一致性分散了指导信号，使其偏离条件信息，留下模糊且不忠实的特征。我们通过一种新颖的信息接地指导机制（IGG）来应对这一挑战，该机制通过注意力机制将指导信号锚定到语义重要的区域。IGG 在采样过程中适应性地增强信息性的 patch，确保指导信号和内容保持紧密对齐。在类条件生成和文本到图像生成任务中，IGG 生成的图像更加清晰、连贯且语义相关，为基于自回归的方法设立了新的标准。 

---
# A Multi-Camera Vision-Based Approach for Fine-Grained Assembly Quality Control 

**Title (ZH)**: 基于多相机视觉的细粒度装配质量控制方法 

**Authors**: Ali Nazeri, Shashank Mishra, Achim Wagner, Martin Ruskowski, Didier Stricker, Jason Rambach  

**Link**: [PDF](https://arxiv.org/pdf/2509.23815)  

**Abstract**: Quality control is a critical aspect of manufacturing, particularly in ensuring the proper assembly of small components in production lines. Existing solutions often rely on single-view imaging or manual inspection, which are prone to errors due to occlusions, restricted perspectives, or lighting inconsistencies. These limitations require the installation of additional inspection stations, which could disrupt the assembly line and lead to increased downtime and costs. This paper introduces a novel multi-view quality control module designed to address these challenges, integrating a multi-camera imaging system with advanced object detection algorithms. By capturing images from three camera views, the system provides comprehensive visual coverage of components of an assembly process. A tailored image fusion methodology combines results from multiple views, effectively resolving ambiguities and enhancing detection reliability. To support this system, we developed a unique dataset comprising annotated images across diverse scenarios, including varied lighting conditions, occlusions, and angles, to enhance applicability in real-world manufacturing environments. Experimental results show that our approach significantly outperforms single-view methods, achieving high precision and recall rates in the identification of improperly fastened small assembly parts such as screws. This work contributes to industrial automation by overcoming single-view limitations, and providing a scalable, cost-effective, and accurate quality control mechanism that ensures the reliability and safety of the assembly line. The dataset used in this study is publicly available to facilitate further research in this domain. 

**Abstract (ZH)**: 多视图质量控制模块在制造中的应用：克服单视角限制，提供可扩展、成本-effective且准确的质量控制机制以确保装配线的可靠性和安全性 

---
# PVTAdpNet: Polyp Segmentation using Pyramid vision transformer with a novel Adapter block 

**Title (ZH)**: PVTAdpNet：基于新型Adapter块的金字塔视觉变换器痔瘇分割方法 

**Authors**: Arshia Yousefi Nezhad, Helia Aghaei, Hedieh Sajedi  

**Link**: [PDF](https://arxiv.org/pdf/2509.23751)  

**Abstract**: Colorectal cancer ranks among the most common and deadly cancers, emphasizing the need for effective early detection and treatment. To address the limitations of traditional colonoscopy, including high miss rates due to polyp variability, we introduce the Pyramid Vision Transformer Adapter Residual Network (PVTAdpNet). This model integrates a U-Net-style encoder-decoder structure with a Pyramid Vision Transformer backbone, novel residual blocks, and adapter-based skip connections. The design enhances feature extraction, dense prediction, and gradient flow, supported by squeeze-and-excitation attention for improved channel-wise feature refinement. PVTAdpNet achieves real-time, accurate polyp segmentation, demonstrating superior performance on benchmark datasets with high mDice and mIoU scores, making it highly suitable for clinical applications. PVTAdpNet obtains a high Dice coefficient of 0.8851 and a mean Intersection over Union (mIoU) of 0.8167 on out-of-distribution polyp datasets. Evaluation of the PolypGen dataset demonstrates PVTAdpNet's capability for real-time, accurate performance within familiar distributions. The source code of our network is available at this https URL 

**Abstract (ZH)**: 结直肠癌是常见且致命的癌症之一，强调了有效早期检测和治疗的必要性。为了解决传统结肠镜检查的局限性，包括由于息肉变异导致的高遗漏率，我们引入了金字塔视觉变换器适配残差网络（PVTAdpNet）。该模型结合了U-Net风格的编码解码结构、金字塔视觉变换器骨干、新型残差块和基于适配器的跳跃连接。该设计增强了特征提取、密集预测和梯度流动，并通过压缩和激励注意力机制提高了通道级特征精炼。PVTAdpNet实现了实时、准确的息肉分割，在基准数据集上表现出色，具有高mDice和mIoU分数，使其非常适合临床应用。PVTAdpNet在未知分布息肉数据集上的Dice系数达到0.8851，平均交并比（mIoU）达到0.8167。PolypGen数据集的评估展示了PVTAdpNet在熟悉分布内的实时、准确性能。我们的网络源代码可在以下链接获取。 

---
# HieraTok: Multi-Scale Visual Tokenizer Improves Image Reconstruction and Generation 

**Title (ZH)**: HieraTok：多尺度视觉分词器提高图像重建和生成 

**Authors**: Cong Chen, Ziyuan Huang, Cheng Zou, Muzhi Zhu, Kaixiang Ji, Jiajia Liu, Jingdong Chen, Hao Chen, Chunhua Shen  

**Link**: [PDF](https://arxiv.org/pdf/2509.23736)  

**Abstract**: In this work, we present HieraTok, a novel multi-scale Vision Transformer (ViT)-based tokenizer that overcomes the inherent limitation of modeling single-scale representations. This is realized through two key designs: (1) multi-scale downsampling applied to the token map generated by the tokenizer encoder, producing a sequence of multi-scale tokens, and (2) a scale-causal attention mechanism that enables the progressive flow of information from low-resolution global semantic features to high-resolution structural details. Coupling these designs, HieraTok achieves significant improvements in both image reconstruction and generation tasks. Under identical settings, the multi-scale visual tokenizer outperforms its single-scale counterpart by a 27.2\% improvement in rFID ($1.47 \rightarrow 1.07$). When integrated into downstream generation frameworks, it achieves a $1.38\times$ faster convergence rate and an 18.9\% boost in gFID ($16.4 \rightarrow 13.3$), which may be attributed to the smoother and more uniformly distributed latent space. Furthermore, by scaling up the tokenizer's training, we demonstrate its potential by a sota rFID of 0.45 and a gFID of 1.82 among ViT tokenizers. To the best of our knowledge, we are the first to introduce multi-scale ViT-based tokenizer in image reconstruction and image generation. We hope our findings and designs advance the ViT-based tokenizers in visual generation tasks. 

**Abstract (ZH)**: 基于多尺度Vision Transformer的HieraTok分词器：在图像重建和生成任务中的应用 

---
# M3DLayout: A Multi-Source Dataset of 3D Indoor Layouts and Structured Descriptions for 3D Generation 

**Title (ZH)**: M3DLayout：多源的3D室内布局及其结构化描述数据集用于3D生成 

**Authors**: Yiheng Zhang, Zhuojiang Cai, Mingdao Wang, Meitong Guo, Tianxiao Li, Li Lin, Yuwang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23728)  

**Abstract**: In text-driven 3D scene generation, object layout serves as a crucial intermediate representation that bridges high-level language instructions with detailed geometric output. It not only provides a structural blueprint for ensuring physical plausibility but also supports semantic controllability and interactive editing. However, the learning capabilities of current 3D indoor layout generation models are constrained by the limited scale, diversity, and annotation quality of existing datasets. To address this, we introduce M3DLayout, a large-scale, multi-source dataset for 3D indoor layout generation. M3DLayout comprises 15,080 layouts and over 258k object instances, integrating three distinct sources: real-world scans, professional CAD designs, and procedurally generated scenes. Each layout is paired with detailed structured text describing global scene summaries, relational placements of large furniture, and fine-grained arrangements of smaller items. This diverse and richly annotated resource enables models to learn complex spatial and semantic patterns across a wide variety of indoor environments. To assess the potential of M3DLayout, we establish a benchmark using a text-conditioned diffusion model. Experimental results demonstrate that our dataset provides a solid foundation for training layout generation models. Its multi-source composition enhances diversity, notably through the Inf3DLayout subset which provides rich small-object information, enabling the generation of more complex and detailed scenes. We hope that M3DLayout can serve as a valuable resource for advancing research in text-driven 3D scene synthesis. 

**Abstract (ZH)**: 基于文本驱动的3D场景生成中，物体布局作为一种关键的中间表示，连接了高层次的语言指令与详细的几何输出。它不仅提供了确保物理合理性的重要结构蓝图，还支持语义可控性和交互编辑。然而，当前3D室内布局生成模型的学习能力受限于现有数据集在规模、多样性和标注质量上的限制。为解决这一问题，我们引入了M3DLayout，这是一个大规模、多来源的3D室内布局生成数据集。M3DLayout包含15,080个布局和超过258k个对象实例，整合了三种不同的来源：真实的扫描数据、专业的CAD设计和程序生成的场景。每个布局都与详尽的结构性文本配合，描述全局场景摘要、大型家具的相对位置以及小型物品的精细布置。这一多样且详细的标注资源使模型能够在广泛多样的室内环境中学习复杂的空间和语义模式。为评估M3DLayout的潜力，我们使用文本条件扩散模型建立了一个基准。实验结果表明，我们的数据集为训练布局生成模型提供了坚实的基础。其多来源的构建增强了多样性，特别通过Inf3DLayout子集提供了丰富的小型物体信息，使生成更加复杂和详细的场景成为可能。我们希望M3DLayout能够成为推进基于文本驱动的3D场景合成研究的重要资源。 

---
# AudioMoG: Guiding Audio Generation with Mixture-of-Guidance 

**Title (ZH)**: AudioMoG: 用混合指导引导音频生成 

**Authors**: Junyou Wang, Zehua Chen, Binjie Yuan, Kaiwen Zheng, Chang Li, Yuxuan Jiang, Jun Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2509.23727)  

**Abstract**: Guidance methods have demonstrated significant improvements in cross-modal audio generation, including text-to-audio (T2A) and video-to-audio (V2A) generation. The popularly adopted method, classifier-free guidance (CFG), steers generation by emphasizing condition alignment, enhancing fidelity but often at the cost of diversity. Recently, autoguidance (AG) has been explored for audio generation, encouraging the sampling to faithfully reconstruct the target distribution and showing increased diversity. Despite these advances, they usually rely on a single guiding principle, e.g., condition alignment in CFG or score accuracy in AG, leaving the full potential of guidance for audio generation untapped. In this work, we explore enriching the composition of the guidance method and present a mixture-of-guidance framework, AudioMoG. Within the design space, AudioMoG can exploit the complementary advantages of distinctive guiding principles by fulfilling their cumulative benefits. With a reduced form, AudioMoG can consider parallel complements or recover a single guiding principle, without sacrificing generality. We experimentally show that, given the same inference speed, AudioMoG approach consistently outperforms single guidance in T2A generation across sampling steps, concurrently showing advantages in V2A, text-to-music, and image generation. These results highlight a "free lunch" in current cross-modal audio generation systems: higher quality can be achieved through mixed guiding principles at the sampling stage without sacrificing inference efficiency. Demo samples are available at: this https URL. 

**Abstract (ZH)**: 指导方法在跨模态音频生成，包括文本到音频（T2A）和视频到音频（V2A）生成中显示出显著改善。尽管广泛采用的方法，无分类器引导（CFG），通过强调条件对齐来引导生成并提高保真度，但通常会牺牲多样性。最近，音频生成中探索了自引导（AG），鼓励采样忠实地重构目标分布并显示出增强的多样性。尽管取得了这些进展，它们通常依赖单一的引导原则，例如CFG中的条件对齐或AG中的分数准确性，从而未能充分利用引导方法的全部潜力。在本文中，我们探讨丰富了指导方法的组成，并提出了一种混合引导框架AudioMoG。在设计空间中，AudioMoG可以利用不同引导原则的互补优势，通过实现其累积效益来发挥这些优势。以简化形式，AudioMoG可以考虑并行补充或恢复单一的引导原则，而不牺牲通用性。实验结果表明，在采样步骤中，与单一引导方法相比，AudioMoG方法在T2A生成中始终表现出更高的性能，并且同样在V2A、文本到音乐和图像生成中显示出优势。这些结果突显了当前跨模态音频生成系统中的“免费午餐”现象：在采样阶段通过混合引导原则可以实现更高质量而不会牺牲推理效率。示范样本可在以下链接获取：this https URL。 

---
# Video Panels for Long Video Understanding 

**Title (ZH)**: 长视频理解的视频面板 

**Authors**: Lars Doorenbos, Federico Spurio, Juergen Gall  

**Link**: [PDF](https://arxiv.org/pdf/2509.23724)  

**Abstract**: Recent Video-Language Models (VLMs) achieve promising results on long-video understanding, but their performance still lags behind that achieved on tasks involving images or short videos. This has led to great interest in improving the long context modeling of VLMs by introducing novel modules and additional complexity. % additional training time. In this paper, we take a different approach: rather than fine-tuning VLMs with the limited data available, we attempt to maximize the performance of existing models. To this end, we propose a novel visual prompting strategy specifically designed for long-video understanding. By combining multiple frames as panels into one image, we effectively trade off spatial details for temporal resolution. Our approach is training-free, parameter-free, and model-agnostic, and can be seamlessly integrated into existing VLMs. Extensive experiments on five established benchmarks across a wide range of model architectures, sizes, and context windows confirm the consistency of our approach. For the TimeScope (Long) dataset, which has the longest videos, the accuracy for video question answering is improved by up to 19.4\%. Overall, our method raises the bar for long video understanding models. We will make our code available upon acceptance. 

**Abstract (ZH)**: 近期的视频-语言模型（VLMs）在长视频理解任务上取得了令人鼓舞的结果，但仍落后于涉及图像或短视频任务的表现。这导致了对提高VLMs的长上下文建模兴趣，通过引入新颖模块和额外复杂性（以及额外训练时间）。在本文中，我们采取不同的方法：而非通过有限的数据微调VLMs，我们试图最大化现有模型的性能。为此，我们提出了一种专门针对长视频理解的新型视觉提示策略。通过将多个帧作为面板合并为一张图像，我们有效权衡了空间细节和时间分辨率。我们的方法无需训练、无需参数，并且具有模型无关性，可以无缝集成到现有VLMs中。跨多种模型架构、规模和上下文窗口的五个标准基准的广泛实验验证了该方法的一致性。对于TimeScope（Long）数据集，该数据集具有最长的视频，视频问答的准确性提高了高达19.4%。总体而言，我们的方法提高了长视频理解模型的标准。接受后我们将提供代码。 

---
# CrimEdit: Controllable Editing for Counterfactual Object Removal, Insertion, and Movement 

**Title (ZH)**: CrimEdit: 可控编辑以实现反事实对象移除、插入和移动 

**Authors**: Boseong Jeon, Junghyuk Lee, Jimin Park, Kwanyoung Kim, Jingi Jung, Sangwon Lee, Hyunbo Shim  

**Link**: [PDF](https://arxiv.org/pdf/2509.23708)  

**Abstract**: Recent works on object removal and insertion have enhanced their performance by handling object effects such as shadows and reflections, using diffusion models trained on counterfactual datasets. However, the performance impact of applying classifier-free guidance to handle object effects across removal and insertion tasks within a unified model remains largely unexplored. To address this gap and improve efficiency in composite editing, we propose CrimEdit, which jointly trains the task embeddings for removal and insertion within a single model and leverages them in a classifier-free guidance scheme -- enhancing the removal of both objects and their effects, and enabling controllable synthesis of object effects during insertion. CrimEdit also extends these two task prompts to be applied to spatially distinct regions, enabling object movement (repositioning) within a single denoising step. By employing both guidance techniques, extensive experiments show that CrimEdit achieves superior object removal, controllable effect insertion, and efficient object movement without requiring additional training or separate removal and insertion stages. 

**Abstract (ZH)**: 近期关于物体删除和插入的研究通过处理物体效果（如阴影和反射）来增强性能，使用了在假设数据集上训练的扩散模型。然而，如何在统一模型中利用无分类器引导技术来处理删除和插入任务中的物体效果，其性能影响仍待探索。为填补这一空白并提高综合编辑的效率，我们提出了CrimEdit，它在单个模型中联合训练删除和插入任务的嵌入，并在其无分类器引导方案中利用这些嵌入——增强物体及其效果的删除，同时在插入过程中实现可控的效果合成。此外，CrimEdit 将这两种任务提示扩展到空间上不同的区域，使物体在单次去噪步骤中实现移动（重新定位）。通过结合使用这两种引导技术，广泛的实验表明，CrimEdit 在物体删除、可控效果插入和高效物体移动方面表现出优越性能，无需额外训练或单独的删除和插入阶段。 

---
# ReWatch-R1: Boosting Complex Video Reasoning in Large Vision-Language Models through Agentic Data Synthesis 

**Title (ZH)**: ReWatch-R1: 通过代理数据合成增强大型视觉语言模型中的复杂视频推理 

**Authors**: Congzhi Zhang, Zhibin Wang, Yinchao Ma, Jiawei Peng, Yihan Wang, Qiang Zhou, Jun Song, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.23652)  

**Abstract**: While Reinforcement Learning with Verifiable Reward (RLVR) significantly advances image reasoning in Large Vision-Language Models (LVLMs), its application to complex video reasoning remains underdeveloped. This gap stems primarily from a critical data bottleneck: existing datasets lack the challenging, multi-hop questions and high-quality, video-grounded Chain-of-Thought (CoT) data necessary to effectively bootstrap RLVR. To address this, we introduce ReWatch, a large-scale dataset built to foster advanced video reasoning. We propose a novel multi-stage synthesis pipeline to synthesize its three components: ReWatch-Caption, ReWatch-QA, and ReWatch-CoT. A core innovation is our Multi-Agent ReAct framework for CoT synthesis, which simulates a human-like "re-watching" process to generate video-grounded reasoning traces by explicitly modeling information retrieval and verification. Building on this dataset, we develop ReWatch-R1 by post-training a strong baseline LVLM with Supervised Fine-Tuning (SFT) and our RLVR framework. This framework incorporates a novel Observation \& Reasoning (O\&R) reward mechanism that evaluates both the final answer's correctness and the reasoning's alignment with video content, directly penalizing hallucination. Our experiments show that ReWatch-R1 achieves state-of-the-art average performance on five challenging video reasoning benchmarks. 

**Abstract (ZH)**: 虽然可验证奖励的强化学习（RLVR）显著推动了大规模视觉-语言模型（LVLMs）中的图像推理，但其在复杂视频推理中的应用仍相对不足。这一差距主要源于一个关键的数据瓶颈：现有数据集缺乏能够有效启动RLVR的具有挑战性和多跳性的问答以及高质量的视频关联思维链（CoT）数据。为了解决这一问题，我们引入了ReWatch，这是一个大型数据集，旨在促进高级视频推理。我们提出了一种新颖的多阶段合成流水线来合成其三个组成部分：ReWatch-Caption、ReWatch-QA和ReWatch-CoT。核心创新是我们提出的多代理ReAct框架用于生成思维链，该框架通过显式建模信息检索和验证来模拟类似人类的“重新观看”过程以生成视频关联的推理轨迹。基于此数据集，我们通过监督微调（SFT）和我们的RLVR框架后训练了一个强大的LVLM，构建了ReWatch-R1。该框架集成了一个新颖的观察与推理（O&R）奖励机制，该机制同时评估最终答案的正确性和推理与视频内容的一致性，并直接惩罚幻觉。我们的实验结果显示，ReWatch-R1在五个具有挑战性的视频推理基准测试中取得了最先进的平均性能。 

---
# BioVessel-Net and RetinaMix: Unsupervised Retinal Vessel Segmentation from OCTA Images 

**Title (ZH)**: BioVessel-Net和RetinaMix：基于OCTA图像的无监督视网膜血管分割 

**Authors**: Cheng Huang, Weizheng Xie, Fan Gao, Yutong Liu, Ruoling Wu, Zeyu Han, Jingxi Qiu, Xiangxiang Wang, Zhenglin Yang, Hao Wang, Yongbin Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.23617)  

**Abstract**: Structural changes in retinal blood vessels are critical biomarkers for the onset and progression of glaucoma and other ocular diseases. However, current vessel segmentation approaches largely rely on supervised learning and extensive manual annotations, which are costly, error-prone, and difficult to obtain in optical coherence tomography angiography. Here we present BioVessel-Net, an unsupervised generative framework that integrates vessel biostatistics with adversarial refinement and a radius-guided segmentation strategy. Unlike pixel-based methods, BioVessel-Net directly models vascular structures with biostatistical coherence, achieving accurate and explainable vessel extraction without labeled data or high-performance computing. To support training and evaluation, we introduce RetinaMix, a new benchmark dataset of 2D and 3D OCTA images with high-resolution vessel details from diverse populations. Experimental results demonstrate that BioVessel-Net achieves near-perfect segmentation accuracy across RetinaMix and existing datasets, substantially outperforming state-of-the-art supervised and semi-supervised methods. Together, BioVessel-Net and RetinaMix provide a label-free, computationally efficient, and clinically interpretable solution for retinal vessel analysis, with broad potential for glaucoma monitoring, blood flow modeling, and progression prediction. Code and dataset are available: this https URL. 

**Abstract (ZH)**: 视网膜血管结构的变化是原发性青光眼和其他眼病发病和进展的关键生物标志物。然而，当前的血管分割方法主要依赖于监督学习和广泛的手动标注，这在光学相干断层扫描血管成像中成本高、易出错且难以实现。这里我们提出BioVessel-Net，这是一种无监督生成框架，该框架结合了血管生物统计学和对抗性细化以及半径引导的分割策略。与基于像素的方法不同，BioVessel-Net 直接使用生物统计学一致性建模血管结构，实现了无标签数据和高性能计算的情况下准确且可解释的血管提取。为了支持训练和评估，我们介绍了RetinaMix，这是一个新的基准数据集，包含来自多元化人群的高分辨率2D和3D OCTA图像。实验结果表明，BioVessel-Net 在RetinaMix 和现有数据集上的分割准确性接近完美，显著优于最先进的监督和半监督方法。BioVessel-Net 和RetinaMix 为无标记、计算高效且临床可解释的视网膜血管分析提供了解决方案，具有广泛的应用潜力，包括青光眼监测、血流建模和进展预测。代码和数据集可获取：this https URL。 

---
# InteractMove: Text-Controlled Human-Object Interaction Generation in 3D Scenes with Movable Objects 

**Title (ZH)**: InteractMove: 文本控制的三维场景中可移动对象的人机交互生成 

**Authors**: Xinhao Cai, Minghang Zheng, Xin Jin, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.23612)  

**Abstract**: We propose a novel task of text-controlled human object interaction generation in 3D scenes with movable objects. Existing human-scene interaction datasets suffer from insufficient interaction categories and typically only consider interactions with static objects (do not change object positions), and the collection of such datasets with movable objects is difficult and costly. To address this problem, we construct the InteractMove dataset for Movable Human-Object Interaction in 3D Scenes by aligning existing human object interaction data with scene contexts, featuring three key characteristics: 1) scenes containing multiple movable objects with text-controlled interaction specifications (including same-category distractors requiring spatial and 3D scene context understanding), 2) diverse object types and sizes with varied interaction patterns (one-hand, two-hand, etc.), and 3) physically plausible object manipulation trajectories. With the introduction of various movable objects, this task becomes more challenging, as the model needs to identify objects to be interacted with accurately, learn to interact with objects of different sizes and categories, and avoid collisions between movable objects and the scene. To tackle such challenges, we propose a novel pipeline solution. We first use 3D visual grounding models to identify the interaction object. Then, we propose a hand-object joint affordance learning to predict contact regions for different hand joints and object parts, enabling accurate grasping and manipulation of diverse objects. Finally, we optimize interactions with local-scene modeling and collision avoidance constraints, ensuring physically plausible motions and avoiding collisions between objects and the scene. Comprehensive experiments demonstrate our method's superiority in generating physically plausible, text-compliant interactions compared to existing approaches. 

**Abstract (ZH)**: 一种控制文本引导可动物体的人机交互生成在3D场景中的新颖任务：InteractMove数据集构建与方法 

---
# Enhancing Polyp Segmentation via Encoder Attention and Dynamic Kernel Update 

**Title (ZH)**: 通过编码器注意力和动态内核更新增强息肉分割 

**Authors**: Fatemeh Salahi Chashmi, Roya Sotoudeh  

**Link**: [PDF](https://arxiv.org/pdf/2509.23502)  

**Abstract**: Polyp segmentation is a critical step in colorectal cancer detection, yet it remains challenging due to the diverse shapes, sizes, and low contrast boundaries of polyps in medical imaging. In this work, we propose a novel framework that improves segmentation accuracy and efficiency by integrating a Dynamic Kernel (DK) mechanism with a global Encoder Attention module. The DK mechanism, initialized by a global context vector from the EA module, iteratively refines segmentation predictions across decoding stages, enabling the model to focus on and accurately delineate complex polyp boundaries. The EA module enhances the network's ability to capture critical lesion features by aggregating multi scale information from all encoder layers. In addition, we employ Unified Channel Adaptation (UCA) in the decoder to standardize feature dimensions across stages, ensuring consistent and computationally efficient information fusion. Our approach extends the lesion-aware kernel framework by introducing a more flexible, attention driven kernel initialization and a unified decoder design. Extensive experiments on the KvasirSEG and CVC ClinicDB benchmark datasets demonstrate that our model outperforms several state of the art segmentation methods, achieving superior Dice and Intersection over Union scores. Moreover, UCA simplifies the decoder structure, reducing computational cost without compromising accuracy. Overall, the proposed method provides a robust and adaptable solution for polyp segmentation, with promising applications in clinical and automated diagnostic systems. 

**Abstract (ZH)**: 一种结合动态内核机制和全局编码器注意力模块的肠道息肉分割框架 

---
# Enhanced Fracture Diagnosis Based on Critical Regional and Scale Aware in YOLO 

**Title (ZH)**: 基于关键区域和尺度aware的YOLO骨折诊断增强方法 

**Authors**: Yuyang Sun, Junchuan Yu, Cuiming Zou  

**Link**: [PDF](https://arxiv.org/pdf/2509.23408)  

**Abstract**: Fracture detection plays a critical role in medical imaging analysis, traditional fracture diagnosis relies on visual assessment by experienced physicians, however the speed and accuracy of this approach are constrained by the expertise. With the rapid advancements in artificial intelligence, deep learning models based on the YOLO framework have been widely employed for fracture detection, demonstrating significant potential in improving diagnostic efficiency and accuracy. This study proposes an improved YOLO-based model, termed Fracture-YOLO, which integrates novel Critical-Region-Selector Attention (CRSelector) and Scale-Aware (ScA) heads to further enhance detection performance. Specifically, the CRSelector module utilizes global texture information to focus on critical features of fracture regions. Meanwhile, the ScA module dynamically adjusts the weights of features at different scales, enhancing the model's capacity to identify fracture targets at multiple scales. Experimental results demonstrate that, compared to the baseline model, Fracture-YOLO achieves a significant improvement in detection precision, with mAP50 and mAP50-95 increasing by 4 and 3, surpassing the baseline model and achieving state-of-the-art (SOTA) performance. 

**Abstract (ZH)**: 骨折检测在医学影像分析中扮演着关键角色，传统骨折诊断依赖经验丰富的医师的视觉评估，然而该方法的速度和准确性受制于医师的经验。随着人工智能的快速发展，基于YOLO框架的深度学习模型被广泛应用于骨折检测，显示了在提高诊断效率和准确性方面的巨大潜力。本研究提出了一种改进的基于YOLO的模型，称为Fracture-YOLO，该模型结合了新型关键区域选择注意力（CRSelector）模块和尺度意识（ScA）头部，以进一步提高检测性能。具体而言，CRSelector模块利用全局纹理信息聚焦骨折区域的关键特征。 Meanwhile, the ScA module dynamically adjusts the weights of features at different scales, enhancing the model's capacity to identify fracture targets at multiple scales. 实验结果表明，与基准模型相比，Fracture-YOLO在检测精度上取得了显著提高，mAP50和mAP50-95分别提高了4和3，超过了基准模型并达到了当前最佳水平（SOTA）。 

---
# Vid-Freeze: Protecting Images from Malicious Image-to-Video Generation via Temporal Freezing 

**Title (ZH)**: Vid-Freeze: 通过时间冻结保护图像免受恶意图像生成为视频的攻击 

**Authors**: Rohit Chowdhury, Aniruddha Bala, Rohan Jaiswal, Siddharth Roheda  

**Link**: [PDF](https://arxiv.org/pdf/2509.23279)  

**Abstract**: The rapid progress of image-to-video (I2V) generation models has introduced significant risks, enabling video synthesis from static images and facilitating deceptive or malicious content creation. While prior defenses such as I2VGuard attempt to immunize images, effective and principled protection to block motion remains underexplored. In this work, we introduce Vid-Freeze - a novel attention-suppressing adversarial attack that adds carefully crafted adversarial perturbations to images. Our method explicitly targets the attention mechanism of I2V models, completely disrupting motion synthesis while preserving semantic fidelity of the input image. The resulting immunized images generate stand-still or near-static videos, effectively blocking malicious content creation. Our experiments demonstrate the impressive protection provided by the proposed approach, highlighting the importance of attention attacks as a promising direction for robust and proactive defenses against misuse of I2V generation models. 

**Abstract (ZH)**: 基于注意力抑制的 Vid-Freeze：一种新颖的图像到视频生成模型对抗攻击 

---
# TRAX: TRacking Axles for Accurate Axle Count Estimation 

**Title (ZH)**: TRAX: 轴跟踪以实现精确轴数估计 

**Authors**: Avinash Rai, Sandeep Jana, Vishal Vijay  

**Link**: [PDF](https://arxiv.org/pdf/2509.23171)  

**Abstract**: Accurate counting of vehicle axles is essential for traffic control, toll collection, and infrastructure development. We present an end-to-end, video-based pipeline for axle counting that tackles limitations of previous works in dense environments. Our system leverages a combination of YOLO-OBB to detect and categorize vehicles, and YOLO to detect tires. Detected tires are intelligently associated to their respective parent vehicles, enabling accurate axle prediction even in complex scenarios. However, there are a few challenges in detection when it comes to scenarios with longer and occluded vehicles. We mitigate vehicular occlusions and partial detections for longer vehicles by proposing a novel TRAX (Tire and Axle Tracking) Algorithm to successfully track axle-related features between frames. Our method stands out by significantly reducing false positives and improving the accuracy of axle-counting for long vehicles, demonstrating strong robustness in real-world traffic videos. This work represents a significant step toward scalable, AI-driven axle counting systems, paving the way for machine vision to replace legacy roadside infrastructure. 

**Abstract (ZH)**: 基于视频的端到端车辆轴数精确计数pipeline及其应用 

---
# CoPatch: Zero-Shot Referring Image Segmentation by Leveraging Untapped Spatial Knowledge in CLIP 

**Title (ZH)**: CoPatch: 通过利用CLIP中未开发的空间知识实现零样本描述图片分割 

**Authors**: Na Min An, Inha Kang, Minhyun Lee, Hyunjung Shim  

**Link**: [PDF](https://arxiv.org/pdf/2509.23098)  

**Abstract**: Spatial grounding is crucial for referring image segmentation (RIS), where the goal of the task is to localize an object described by language. Current foundational vision-language models (VLMs), such as CLIP, excel at aligning images and text but struggle with understanding spatial relationships. Within the language stream, most existing methods often focus on the primary noun phrase when extracting local text features, undermining contextual tokens. Within the vision stream, CLIP generates similar features for images with different spatial layouts, resulting in limited sensitivity to spatial structure. To address these limitations, we propose \textsc{CoPatch}, a zero-shot RIS framework that leverages internal model components to enhance spatial representations in both text and image modalities. For language, \textsc{CoPatch} constructs hybrid text features by incorporating context tokens carrying spatial cues. For vision, it extracts patch-level image features using our novel path discovered from intermediate layers, where spatial structure is better preserved. These enhanced features are fused into a clustered image-text similarity map, \texttt{CoMap}, enabling precise mask selection. As a result, \textsc{CoPatch} significantly improves spatial grounding in zero-shot RIS across RefCOCO, RefCOCO+, RefCOCOg, and PhraseCut (+ 2--7 mIoU) without requiring any additional training. Our findings underscore the importance of recovering and leveraging the untapped spatial knowledge inherently embedded in VLMs, thereby paving the way for opportunities in zero-shot RIS. 

**Abstract (ZH)**: 空间接地是引用图像分割（RIS）的关键任务，其中目标是定位由语言描述的物体。现有的基础视觉-语言模型（VLMs），如CLIP，擅长图像和文本的对齐，但在理解空间关系方面存在困难。在语言流中，现有方法大多在提取局部文本特征时关注主要名词短语，忽视了上下文词汇。在视觉流中，CLIP对具有不同空间布局的图像生成相似的特征，导致对空间结构的敏感性有限。为解决这些局限性，我们提出了\textsc{CoPatch}，这是一种零样本RIS框架，利用模型内部组件增强文本和图像模态中的空间表示。对于语言，\textsc{CoPatch}通过结合携带空间线索的上下文词汇构建混合文本特征。对于视觉，它使用从中间层发现的新型路径提取补丁级图像特征，其中空间结构得到更好地保持。这些增强的特征被融合进集群图像-文本相似性图\texttt{CoMap}中，可以实现精确的掩码选择。结果，\textsc{CoPatch}在零样本RIS中显著改善了空间接地（在RefCOCO、RefCOCO+、RefCOCOg和PhraseCut上分别提高2-7个mIoU点）而不需额外训练。我们的研究结果强调了恢复和利用VLMs中固有嵌入的未充分利用的空间知识的重要性，从而为零样本RIS提供了新的机遇。 

---
# GeLoc3r: Enhancing Relative Camera Pose Regression with Geometric Consistency Regularization 

**Title (ZH)**: GeLoc3r: 通过几何一致性正则化提高相对相机姿态回归性能 

**Authors**: Jingxing Li, Yongjae Lee, Deliang Fan  

**Link**: [PDF](https://arxiv.org/pdf/2509.23038)  

**Abstract**: Prior ReLoc3R achieves breakthrough performance with fast 25ms inference and state-of-the-art regression accuracy, yet our analysis reveals subtle geometric inconsistencies in its internal representations that prevent reaching the precision ceiling of correspondence-based methods like MASt3R (which require 300ms per pair). In this work, we present GeLoc3r, a novel approach to relative camera pose estimation that enhances pose regression methods through Geometric Consistency Regularization (GCR). GeLoc3r overcomes the speed-accuracy dilemma by training regression networks to produce geometrically consistent poses without inference-time geometric computation. During training, GeLoc3r leverages ground-truth depth to generate dense 3D-2D correspondences, weights them using a FusionTransformer that learns correspondence importance, and computes geometrically-consistent poses via weighted RANSAC. This creates a consistency loss that transfers geometric knowledge into the regression network. Unlike FAR method which requires both regression and geometric solving at inference, GeLoc3r only uses the enhanced regression head at test time, maintaining ReLoc3R's fast speed and approaching MASt3R's high accuracy. On challenging benchmarks, GeLoc3r consistently outperforms ReLoc3R, achieving significant improvements including 40.45% vs. 34.85% AUC@5° on the CO3Dv2 dataset (16% relative improvement), 68.66% vs. 66.70% AUC@5° on RealEstate10K, and 50.45% vs. 49.60% on MegaDepth1500. By teaching geometric consistency during training rather than enforcing it at inference, GeLoc3r represents a paradigm shift in how neural networks learn camera geometry, achieving both the speed of regression and the geometric understanding of correspondence methods. 

**Abstract (ZH)**: GeLoc3r：通过几何一致正则化增强的姿态回归方法在相对相机姿态估计中的应用 

---
# Soft-Di[M]O: Improving One-Step Discrete Image Generation with Soft Embeddings 

**Title (ZH)**: Soft-Di[M]O: 基于软嵌入改进的一步离散图像生成 

**Authors**: Yuanzhi Zhu, Xi Wang, Stéphane Lathuilière, Vicky Kalogeiton  

**Link**: [PDF](https://arxiv.org/pdf/2509.22925)  

**Abstract**: One-step generators distilled from Masked Diffusion Models (MDMs) compress multiple sampling steps into a single forward pass, enabling efficient text and image synthesis. However, they suffer two key limitations: they inherit modeling bias from the teacher, and their discrete token outputs block gradient flow, preventing post-distillation refinements such as adversarial training, reward-based fine-tuning, and Test-Time Embedding Optimization (TTEO). In this work, we introduce soft embeddings, a simple relaxation that replaces discrete tokens with the expected embeddings under the generator's output distribution. Soft embeddings preserve representation fidelity for one-step discrete generator while providing a fully differentiable continuous surrogate that is compatible with teacher backbones and tokenizer decoders. Integrating soft embeddings into the Di[M]O distillation framework (denoted Soft-Di[M]O) makes one-step generators end-to-end trainable and enables straightforward application of GAN-based refinement, differentiable reward fine-tuning, and TTEO. Empirically, across multiple MDM teachers (e.g., MaskBit, MaskGen), Soft-Di[M]O achieves state-of-the-art one-step results: improved class-to-image performance, a one-step FID of 1.56 on ImageNet-256 with GAN-based refinement, along with higher GenEval and HPS scores on text-to-image with reward fine-tuning, and further gains from TTEO. 

**Abstract (ZH)**: 一步生成器：从掩码扩散模型衍生的软嵌入表示的一站式图像生成方法 

---
# TY-RIST: Tactical YOLO Tricks for Real-time Infrared Small Target Detection 

**Title (ZH)**: TY-RIST: 战术YOLO技巧用于实时红外小型目标检测 

**Authors**: Abdulkarim Atrash, Omar Moured, Yufan Chen, Jiaming Zhang, Seyda Ertekin, Omur Ugur  

**Link**: [PDF](https://arxiv.org/pdf/2509.22909)  

**Abstract**: Infrared small target detection (IRSTD) is critical for defense and surveillance but remains challenging due to (1) target loss from minimal features, (2) false alarms in cluttered environments, (3) missed detections from low saliency, and (4) high computational costs. To address these issues, we propose TY-RIST, an optimized YOLOv12n architecture that integrates (1) a stride-aware backbone with fine-grained receptive fields, (2) a high-resolution detection head, (3) cascaded coordinate attention blocks, and (4) a branch pruning strategy that reduces computational cost by about 25.5% while marginally improving accuracy and enabling real-time inference. We also incorporate the Normalized Gaussian Wasserstein Distance (NWD) to enhance regression stability. Extensive experiments on four benchmarks and across 20 different models demonstrate state-of-the-art performance, improving mAP at 0.5 IoU by +7.9%, Precision by +3%, and Recall by +10.2%, while achieving up to 123 FPS on a single GPU. Cross-dataset validation on a fifth dataset further confirms strong generalization capability. Additional results and resources are available at this https URL 

**Abstract (ZH)**: 红外小目标检测（IRSTD）对于防御和监控至关重要，但由于（1）微弱特征导致的目标丢失，（2）杂波环境下误报警，（3）低显着性导致的漏检，以及（4）高计算成本，仍具有挑战性。为了解决这些问题，我们提出了TY-RIST，这是一种优化的YOLOv12n架构，结合了（1）具有精细感受野的步幅感知骨干网，（2）高分辨率检测头，（3）级联坐标注意力块，以及（4）分支剪枝策略，该策略在计算成本减少约25.5%的同时，轻微提高了准确率并支持实时推理。我们还将规范化高斯 Wasserstein 距离（NWD）纳入以增强回归稳定性。在四个基准上的广泛实验以及20种不同模型的测试表明，TY-RIST 达到了最先进的性能，mAP在0.5 IoU下的提升幅度为+7.9%，精确率提升+3%，召回率提升+10.2%，且在单个GPU上实现了高达123 FPS。跨数据集验证在第五个数据集上进一步证实了强泛化能力。更多信息和资源请访问此网址。 

---
# Convolutional Set Transformer 

**Title (ZH)**: 卷积集转换器 

**Authors**: Federico Chinello, Giacomo Boracchi  

**Link**: [PDF](https://arxiv.org/pdf/2509.22889)  

**Abstract**: We introduce the Convolutional Set Transformer (CST), a novel neural architecture designed to process image sets of arbitrary cardinality that are visually heterogeneous yet share high-level semantics - such as a common category, scene, or concept. Existing set-input networks, e.g., Deep Sets and Set Transformer, are limited to vector inputs and cannot directly handle 3D image tensors. As a result, they must be cascaded with a feature extractor, typically a CNN, which encodes images into embeddings before the set-input network can model inter-image relationships. In contrast, CST operates directly on 3D image tensors, performing feature extraction and contextual modeling simultaneously, thereby enabling synergies between the two processes. This design yields superior performance in tasks such as Set Classification and Set Anomaly Detection and further provides native compatibility with CNN explainability methods such as Grad-CAM, unlike competing approaches that remain opaque. Finally, we show that CSTs can be pre-trained on large-scale datasets and subsequently adapted to new domains and tasks through standard Transfer Learning schemes. To support further research, we release CST-15, a CST backbone pre-trained on ImageNet (this https URL). 

**Abstract (ZH)**: 我们介绍了Convolutional Set Transformer (CST)，这是一种新型的神经架构，设计用于处理具有任意基数且在视觉上异构但共享高级语义（如共同类别、场景或概念）的图像集合。现有的集输入网络，例如Deep Sets和Set Transformer，仅限于处理向量输入，无法直接处理3D图像张量。因此，它们必须与特征提取器（通常为CNN）级联，将图像编码为嵌入向量，才能使集输入网络能够建模图像之间的关系。相比之下，CST可以直接操作3D图像张量，同时进行特征提取和上下文建模，从而在两个过程之间实现协同效应。这种设计在集分类和集异常检测等任务中表现出了优越的性能，并进一步提供了与CNN解释性方法（如Grad-CAM）的原生兼容性，而竞争方法仍然具有不透明性。最后，我们展示了CST可以在大规模数据集上进行预训练，并通过标准的迁移学习方案适应新的领域和任务。为了支持进一步的研究，我们发布了基于ImageNet预训练的CST-15骨干网络（详见此处：this https URL）。 

---
# Multimodal Slice Interaction Network Enhanced by Transfer Learning for Precise Segmentation of Internal Gross Tumor Volume in Lung Cancer PET/CT Imaging 

**Title (ZH)**: 基于迁移学习增强的多模态切片交互网络在肺癌PET/CT成像中精确分割内部粗大肿瘤体积 

**Authors**: Yi Luo, Yike Guo, Hamed Hooshangnejad, Rui Zhang, Xue Feng, Quan Chen, Wil Ngwa, Kai Ding  

**Link**: [PDF](https://arxiv.org/pdf/2509.22841)  

**Abstract**: Lung cancer remains the leading cause of cancerrelated deaths globally. Accurate delineation of internal gross tumor volume (IGTV) in PET/CT imaging is pivotal for optimal radiation therapy in mobile tumors such as lung cancer to account for tumor motion, yet is hindered by the limited availability of annotated IGTV datasets and attenuated PET signal intensity at tumor boundaries. In this study, we present a transfer learningbased methodology utilizing a multimodal interactive perception network with MAMBA, pre-trained on extensive gross tumor volume (GTV) datasets and subsequently fine-tuned on a private IGTV cohort. This cohort constitutes the PET/CT subset of the Lung-cancer Unified Cross-modal Imaging Dataset (LUCID). To further address the challenge of weak PET intensities in IGTV peripheral slices, we introduce a slice interaction module (SIM) within a 2.5D segmentation framework to effectively model inter-slice relationships. Our proposed module integrates channel and spatial attention branches with depthwise convolutions, enabling more robust learning of slice-to-slice dependencies and thereby improving overall segmentation performance. A comprehensive experimental evaluation demonstrates that our approach achieves a Dice of 0.609 on the private IGTV dataset, substantially surpassing the conventional baseline score of 0.385. This work highlights the potential of transfer learning, coupled with advanced multimodal techniques and a SIM to enhance the reliability and clinical relevance of IGTV segmentation for lung cancer radiation therapy planning. 

**Abstract (ZH)**: 基于迁移学习的多模态交互感知网络在肺 cancer 内在粗略肿瘤体积分割中的应用：挑战及解决方案 

---
# Seeing Isn't Believing: Context-Aware Adversarial Patch Synthesis via Conditional GAN 

**Title (ZH)**: 看见并不等于相信：基于条件GAN的上下文感知对抗性补丁合成 

**Authors**: Roie Kazoom, Alon Goldberg, Hodaya Cohen, Ofer Hadar  

**Link**: [PDF](https://arxiv.org/pdf/2509.22836)  

**Abstract**: Adversarial patch attacks pose a severe threat to deep neural networks, yet most existing approaches rely on unrealistic white-box assumptions, untargeted objectives, or produce visually conspicuous patches that limit real-world applicability. In this work, we introduce a novel framework for fully controllable adversarial patch generation, where the attacker can freely choose both the input image x and the target class y target, thereby dictating the exact misclassification outcome. Our method combines a generative U-Net design with Grad-CAM-guided patch placement, enabling semantic-aware localization that maximizes attack effectiveness while preserving visual realism. Extensive experiments across convolutional networks (DenseNet-121, ResNet-50) and vision transformers (ViT-B/16, Swin-B/16, among others) demonstrate that our approach achieves state-of-the-art performance across all settings, with attack success rates (ASR) and target-class success (TCS) consistently exceeding 99%.
Importantly, we show that our method not only outperforms prior white-box attacks and untargeted baselines, but also surpasses existing non-realistic approaches that produce detectable artifacts. By simultaneously ensuring realism, targeted control, and black-box applicability-the three most challenging dimensions of patch-based attacks-our framework establishes a new benchmark for adversarial robustness research, bridging the gap between theoretical attack strength and practical stealthiness. 

**Abstract (ZH)**: 对抗补丁攻击对深度神经网络构成了严重威胁，但大多数现有方法依赖于不切实际的白盒假设、非目标攻击目标或产生视觉上显眼的补丁，限制了其实用性。在本工作中，我们引入了一种全新的可完全控制的对抗补丁生成框架，攻击者可以自由选择输入图像x和目标类别y，从而决定具体的分类错误结果。我们的方法结合了生成的U-Net设计与Grad-CAM指导的补丁放置，实现了语义意识的定位，最大化攻击效果同时保留视觉真实性。在卷积网络（DenseNet-121、ResNet-50）和视觉变压器（ViT-B/16、Swin-B/16等）上的广泛实验表明，我们的方法在所有设置中均达到最优性能，攻击成功率（ASR）和目标类成功（TCS）一致地超过99%。尤为重要的是，我们证明了我们的方法不仅优于之前的白盒攻击和非目标基准，还超越了现有不可靠的方法，这些方法会产生可检测的伪影。通过同时确保现实性、目标控制和黑盒适用性——补丁攻击最具有挑战性的三个维度——我们的框架为对抗鲁棒性研究建立了新的基准，填补了理论攻击强度与实际隐蔽性之间的差距。 

---
# VideoScore2: Think before You Score in Generative Video Evaluation 

**Title (ZH)**: VideoScore2: 评分之前先思考 在生成视频评估中的应用 

**Authors**: Xuan He, Dongfu Jiang, Ping Nie, Minghao Liu, Zhengxuan Jiang, Mingyi Su, Wentao Ma, Junru Lin, Chun Ye, Yi Lu, Keming Wu, Benjamin Schneider, Quy Duc Do, Zhuofeng Li, Yiming Jia, Yuxuan Zhang, Guo Cheng, Haozhe Wang, Wangchunshu Zhou, Qunshu Lin, Yuanxing Zhang, Ge Zhang, Wenhao Huang, Wenhu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.22799)  

**Abstract**: Recent advances in text-to-video generation have produced increasingly realistic and diverse content, yet evaluating such videos remains a fundamental challenge due to their multi-faceted nature encompassing visual quality, semantic alignment, and physical consistency. Existing evaluators and reward models are limited to single opaque scores, lack interpretability, or provide only coarse analysis, making them insufficient for capturing the comprehensive nature of video quality assessment. We present VideoScore2, a multi-dimensional, interpretable, and human-aligned framework that explicitly evaluates visual quality, text-to-video alignment, and physical/common-sense consistency while producing detailed chain-of-thought rationales. Our model is trained on a large-scale dataset VideoFeedback2 containing 27,168 human-annotated videos with both scores and reasoning traces across three dimensions, using a two-stage pipeline of supervised fine-tuning followed by reinforcement learning with Group Relative Policy Optimization (GRPO) to enhance analytical robustness. Extensive experiments demonstrate that VideoScore2 achieves superior performance with 44.35 (+5.94) accuracy on our in-domain benchmark VideoScore-Bench-v2 and 50.37 (+4.32) average performance across four out-of-domain benchmarks (VideoGenReward-Bench, VideoPhy2, etc), while providing interpretable assessments that bridge the gap between evaluation and controllable generation through effective reward modeling for Best-of-N sampling. Project Page: this https URL 

**Abstract (ZH)**: Recent advances in text-to-video generation have produced increasingly realistic and diverse content, yet evaluating such videos remains a fundamental challenge due to their multi-faceted nature encompassing visual quality, semantic alignment, and physical consistency. Existing evaluators and reward models are limited to single opaque scores, lack interpretability, or provide only coarse analysis, making them insufficient for capturing the comprehensive nature of video quality assessment. We present VideoScore2, a multi-dimensional, interpretable, and human-aligned framework that explicitly evaluates visual quality, text-to-video alignment, and physical/common-sense consistency while producing detailed chain-of-thought rationales. Our model is trained on a large-scale dataset VideoFeedback2 containing 27,168 human-annotated videos with both scores and reasoning traces across three dimensions, using a two-stage pipeline of supervised fine-tuning followed by reinforcement learning with Group Relative Policy Optimization (GRPO) to enhance analytical robustness. Extensive experiments demonstrate that VideoScore2 achieves superior performance with 44.35 (+5.94) accuracy on our in-domain benchmark VideoScore-Bench-v2 and 50.37 (+4.32) average performance across four out-of-domain benchmarks (VideoGenReward-Bench, VideoPhy2, etc), while providing interpretable assessments that bridge the gap between evaluation and controllable generation through effective reward modeling for Best-of-N sampling. Project Page: [this https URL] 

---
# UESA-Net: U-Shaped Embedded Multidirectional Shrinkage Attention Network for Ultrasound Nodule Segmentation 

**Title (ZH)**: UESA-Net: U型嵌入多方向收缩注意力网络-ultrasound结节分割 

**Authors**: Tangqi Shi, Pietro Lio  

**Link**: [PDF](https://arxiv.org/pdf/2509.22763)  

**Abstract**: Background: Breast and thyroid cancers pose an increasing public-health burden. Ultrasound imaging is a cost-effective, real-time modality for lesion detection and segmentation, yet suffers from speckle noise, overlapping structures, and weak global-local feature interactions. Existing networks struggle to reconcile high-level semantics with low-level spatial details. We aim to develop a segmentation framework that bridges the semantic gap between global context and local detail in noisy ultrasound images.
Methods: We propose UESA-Net, a U-shaped network with multidirectional shrinkage attention. The encoder-decoder architecture captures long-range dependencies and fine-grained structures of lesions. Within each encoding block, attention modules operate along horizontal, vertical, and depth directions to exploit spatial details, while a shrinkage (threshold) strategy integrates prior knowledge and local features. The decoder mirrors the encoder but applies a pairwise shrinkage mechanism, combining prior low-level physical cues with corresponding encoder features to enhance context modeling.
Results: On two public datasets - TN3K (3493 images) and BUSI (780 images) - UESA-Net achieved state-of-the-art performance with intersection-over-union (IoU) scores of 0.8487 and 0.6495, respectively.
Conclusions: UESA-Net effectively aggregates multidirectional spatial information and prior knowledge to improve robustness and accuracy in breast and thyroid ultrasound segmentation, demonstrating superior performance to existing methods on multiple benchmarks. 

**Abstract (ZH)**: 背景：乳腺癌和甲状腺癌对公共健康造成日益增大的负担。超声成像是一种成本有效且实时的病变检测和分割方法，但受到斑点噪声、重叠结构和全局-局部特征交互微弱的限制。现有网络难以平衡高层语义与低层空间细节。本文旨在开发一种分割框架，以在噪声超声图像中弥合全局上下文与局部细节之间的语义差距。

方法：我们提出了UESA-Net，这是一种具有多方向收缩注意机制的U形网络。编码-解码架构捕捉到病变的长距离依赖性和精细结构。在每个编码块中，注意力模块沿横向、纵向和深度方向操作，以利用空间细节，而收缩（阈值）策略则结合了先验知识和局部特征。解码器镜像编码器结构，但应用成对收缩机制，结合先验低级物理线索和相应的编码特征以增强上下文建模。

结果：在两个公开数据集TN3K（3493张图像）和BUSI（780张图像）上，UESA-Net分别实现了交并比（IoU）值0.8487和0.6495，达到了最先进的性能。

结论：UESA-Net有效整合了多方向空间信息和先验知识，提高了乳腺和甲状腺超声分割的鲁棒性和准确性，展示了在多个基准上优于现有方法的性能。 

---
# Learning What To Hear: Boosting Sound-Source Association For Robust Audiovisual Instance Segmentation 

**Title (ZH)**: 学习听见什么：提升稳健的音视频实例分割中的声源关联 

**Authors**: Jinbae Seo, Hyeongjun Kwon, Kwonyoung Kim, Jiyoung Lee, Kwanghoon Sohn  

**Link**: [PDF](https://arxiv.org/pdf/2509.22740)  

**Abstract**: Audiovisual instance segmentation (AVIS) requires accurately localizing and tracking sounding objects throughout video sequences. Existing methods suffer from visual bias stemming from two fundamental issues: uniform additive fusion prevents queries from specializing to different sound sources, while visual-only training objectives allow queries to converge to arbitrary salient objects. We propose Audio-Centric Query Generation using cross-attention, enabling each query to selectively attend to distinct sound sources and carry sound-specific priors into visual decoding. Additionally, we introduce Sound-Aware Ordinal Counting (SAOC) loss that explicitly supervises sounding object numbers through ordinal regression with monotonic consistency constraints, preventing visual-only convergence during training. Experiments on AVISeg benchmark demonstrate consistent improvements: +1.64 mAP, +0.6 HOTA, and +2.06 FSLA, validating that query specialization and explicit counting supervision are crucial for accurate audiovisual instance segmentation. 

**Abstract (ZH)**: 音频中心查询生成与音觉感知计数在音视频实例分割中的应用 

---
# IBiT: Utilizing Inductive Biases to Create a More Data Efficient Attention Mechanism 

**Title (ZH)**: IBiT：利用归纳偏置创建一种更数据高效的关注机制 

**Authors**: Adithya Giri  

**Link**: [PDF](https://arxiv.org/pdf/2509.22719)  

**Abstract**: In recent years, Transformer-based architectures have become the dominant method for Computer Vision applications. While Transformers are explainable and scale well with dataset size, they lack the inductive biases of Convolutional Neural Networks. While these biases may be learned on large datasets, we show that introducing these inductive biases through learned masks allow Vision Transformers to learn on much smaller datasets without Knowledge Distillation. These Transformers, which we call Inductively Biased Image Transformers (IBiT), are significantly more accurate on small datasets, while retaining the explainability Transformers. 

**Abstract (ZH)**: 基于诱导偏置的图像变压器：使用学习掩模在小数据集上学习的计算机视觉变换器 

---
# GZSL-MoE: Apprentissage G{é}n{é}ralis{é} Z{é}ro-Shot bas{é} sur le M{é}lange d'Experts pour la Segmentation S{é}mantique de Nuages de Points 3DAppliqu{é} {à} un Jeu de Donn{é}es d'Environnement de Collaboration Humain-Robot 

**Title (ZH)**: GZSL-MoE: 专家混合基于零-shot 通用学习的点云语义分割方法及其在人类-机器人协作环境数据集上的应用 

**Authors**: Ahed Alboody  

**Link**: [PDF](https://arxiv.org/pdf/2509.22708)  

**Abstract**: Generative Zero-Shot Learning approach (GZSL) has demonstrated significant potential in 3D point cloud semantic segmentation tasks. GZSL leverages generative models like GANs or VAEs to synthesize realistic features (real features) of unseen classes. This allows the model to label unseen classes during testing, despite being trained only on seen classes. In this context, we introduce the Generalized Zero-Shot Learning based-upon Mixture-of-Experts (GZSL-MoE) model. This model incorporates Mixture-of-Experts layers (MoE) to generate fake features that closely resemble real features extracted using a pre-trained KPConv (Kernel Point Convolution) model on seen classes. The main contribution of this paper is the integration of Mixture-of-Experts into the Generator and Discriminator components of the Generative Zero-Shot Learning model for 3D point cloud semantic segmentation, applied to the COVERED dataset (CollabOratiVE Robot Environment Dataset) for Human-Robot Collaboration (HRC) environments. By combining the Generative Zero-Shot Learning model with Mixture-of- Experts, GZSL-MoE for 3D point cloud semantic segmentation provides a promising solution for understanding complex 3D environments, especially when comprehensive training data for all object classes is unavailable. The performance evaluation of the GZSL-MoE model highlights its ability to enhance performance on both seen and unseen classes. Keywords Generalized Zero-Shot Learning (GZSL), 3D Point Cloud, 3D Semantic Segmentation, Human-Robot Collaboration, COVERED (CollabOratiVE Robot Environment Dataset), KPConv, Mixture-of Experts 

**Abstract (ZH)**: 基于混合专家的泛化零样本学习方法（GZSL-MoE）在3D点云语义分割任务中的应用 

---
# YOLO-based Bearing Fault Diagnosis With Continuous Wavelet Transform 

**Title (ZH)**: 基于YOLO的连续小波变换轴承故障诊断 

**Authors**: Po-Heng Chou, Wei-Lung Mao, Ru-Ping Lin  

**Link**: [PDF](https://arxiv.org/pdf/2509.03070)  

**Abstract**: This letter proposes a YOLO-based framework for spatial bearing fault diagnosis using time-frequency spectrograms derived from continuous wavelet transform (CWT). One-dimensional vibration signals are first transformed into time-frequency spectrograms using Morlet wavelets to capture transient fault signatures. These spectrograms are then processed by YOLOv9, v10, and v11 models to classify fault types. Evaluated on three benchmark datasets, including Case Western Reserve University (CWRU), Paderborn University (PU), and Intelligent Maintenance System (IMS), the proposed CWT-YOLO pipeline achieves significantly higher accuracy and generalizability than the baseline MCNN-LSTM model. Notably, YOLOv11 reaches mAP scores of 99.4% (CWRU), 97.8% (PU), and 99.5% (IMS). In addition, its region-aware detection mechanism enables direct visualization of fault locations in spectrograms, offering a practical solution for condition monitoring in rotating machinery. 

**Abstract (ZH)**: 基于连续小波变换的时间-frequency光谱图的YOLO架构的空间轴承故障诊断方法 

---
