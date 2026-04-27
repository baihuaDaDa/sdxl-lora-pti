# 基于 SDXL 的 LoRA 微调、PTI 训练与 LoRA Merge 实验报告
> ACM Class 2023 - 张博涵；学号：523030910144

## 摘要

本报告围绕课程作业“扩散模型的微调”要求，对仓库中的 SDXL LoRA 训练、PTI 训练、推理与 LoRA merge 流程进行了系统实验与分析。相较于上一版仅聚焦单一方法对比，本版报告进一步补充了 4 种 PTI 风格与原模型的直接对比、4 种风格之间的相互 transfer 实验、full body 与 heavy background 的 prompt 调优实验，以及 2 路、3 路、4 路 merge 的分层对比，从而更全面地体现最终训练成果。整个报告共整理了 9 个实验板块、139 张生成图像，覆盖 mygo、jojo、wushan、cyberpunk 四类风格，且大多数对比均采用多 seed 设置以降低偶然性。

整体结果表明，本次课程作业的完成度较高，训练结果也具有较强可展示性。PTI 在 4 种风格上均能稳定优于原始 SDXL 的“仅靠 prompt 触发”的生成结果，尤其在风格绑定、角色特征一致性与 token 可控性方面优势明显。以 mygo 为代表的主实验中，2000 步训练配合 0.8 的 LoRA scale 给出了最稳健的综合效果；对 cyberpunk 风格而言，portrait + scene 的混合数据集显著提升了复杂背景提示词的响应能力；对 style transfer 而言，jojo 与 cyberpunk 的迁移效果最强，mygo 与 wushan 则呈现更温和但仍清晰可辨的风格注入。LoRA merge 方面，2 路 merge 明显优于 4 路 merge，说明“少量风格合并可行，多风格直接累加风险较大”。full body prompt 在经过显式构图约束和长宽比调整后已有明显改善，但 heavy background 的控制仍然相对困难。

## 1. 任务概述与完成情况

根据课程要求，个人报告应至少覆盖数据集选择、模型训练过程、生成结果与过程分析。本次工作已经完整覆盖上述要求，并且在实验深度上超出了“跑通一个训练脚本并展示若干样图”的最低标准，具体体现在以下几方面：

1. 完成了 4 组 PTI SDXL 风格 LoRA 的系统实验，并复用 1 组 DreamBooth SDXL LoRA 作为方法比较基线。
2. 完成了训练过程观察、不同 checkpoint 与不同推理 scale 的扫描、不同训练集构成对比、不同 prompt 能力分析、style transfer、LoRA merge 等多维度实验。
3. 共整理 9 个实验板块、139 张生成图像，绝大多数实验采用多 seed 对比，保证结论不依赖单一幸运样本。
4. 最终结果不仅能说明“模型成功训练”，还能够说明“训练后的模型在何处显著优于原模型、何处具有扩展能力、何处会出现局限”。

因此，从课程作业“任务完成度”和“报告质量”两个维度看，本次工作已经达到较高完成水平。

## 2. 数据集、模型与实验设置

### 2.1 数据集

仓库中同时存在 data 与 data_bohan 两套同名数据目录。现有训练脚本实际指向 data 目录；检查后发现 5 个同名子目录在两套目录中的文件数一致，因此本报告在数据描述上以 data_bohan 的语义组织为主，在实现层面与 data 目录对应。

| 风格 | 数据目录 | 图像数 | 说明 |
| --- | --- | ---: | --- |
| mygo | data_bohan/mygo | 25 | 动漫人物外观与舞台演唱场景 |
| jojo | data_bohan/jojo | 25 | 夸张肌肉线条、强烈阴影与姿态风格 |
| wushan | data_bohan/wushan | 25 | 水墨武侠、火焰兵器、动态战斗姿态 |
| cyberpunk_portrait | data_bohan/cyberpunk_portrait | 15 | 纯 portrait 风格，强调角色近景 |
| cyberpunk_all | data_bohan/cyberpunk_all | 25 | 15 张 portrait + 10 张 urban scene，强调角色与赛博城市背景共同出现 |

### 2.2 训练配置

#### PTI SDXL 配置

依据 [training_scripts/run_lora_pti_sdxl_single.sh](../training_scripts/run_lora_pti_sdxl_single.sh) 及同系列脚本，PTI 训练采用如下核心设置：

| 项目 | 设置 |
| --- | --- |
| 基座模型 | SDXL base 1.0 |
| 分辨率 | 1024 x 1024 |
| batch size | 1 |
| gradient accumulation | 4 |
| LoRA rank | 16 |
| 最大训练步数 | 2000 |
| 保存间隔 | 200 步 |
| LoRA 学习率 | 1e-4 |
| text encoder LoRA 学习率 | 5e-6 |
| TI 学习率 | 5e-4 |
| 调度器 | constant |
| 两阶段策略 | 0 到 500 步只训练 TI；500 到 2000 步训练 LoRA |

PTI 训练脚本本体位于 [training_scripts/train_lora_w_ti_sdxl.py](../training_scripts/train_lora_w_ti_sdxl.py)。该脚本会同时向 SDXL 双文本编码器注册 placeholder token 与 learned embedding，这也是后续 merge 实验中 token 触发仍然成立的根本原因。

#### DreamBooth SDXL 配置

DreamBooth 参考脚本为 [training_scripts/run_lora_sdxl.sh](../training_scripts/run_lora_sdxl.sh)。与 PTI 相比，主要差别是：

1. 使用固定 instance prompt，而不是显式的可学习 placeholder token。
2. 学习率调度采用 cosine，warmup 为 100 步。
3. 当前工作区中可直接复用的 DreamBooth SDXL 权重以 mygo 风格为主，因此方法对比实验主要聚焦 mygo。

### 2.3 推理设置

除专门的 prompt 调优实验外，大部分推理统一采用以下设置：

| 项目 | 设置 |
| --- | --- |
| 推理步数 | 40 |
| guidance scale | 7.5 |
| 分辨率 | 1024 x 1024 |
| 负面提示词 | low quality, blurry, deformed, extra fingers, bad anatomy, watermark, text |
| 多 seed | 13、37、73；部分 merge 实验为 13、37 |

批量实验规格文件为 [report_assets/visual_report_experiments.json](../report_assets/visual_report_experiments.json)，批量运行脚本为 [scripts/run_sdxl_report_experiments.py](../scripts/run_sdxl_report_experiments.py)。

## 3. 训练过程观察

### 3.1 PTI 两阶段训练的稳定性

下图给出了 mygo、cyberpunk_portrait、cyberpunk_all 三个 PTI 训练日志解析得到的移动平均损失曲线。灰色虚线对应第 500 步，即 TI-only 阶段切换到 LoRA 训练阶段的位置。

![PTI loss](plots/pti_training_loss_curves.png)

可以观察到：

1. 三条曲线在 500 步切换点附近都没有出现明显发散，说明 TI 到 LoRA 的切换是稳定的。
2. 在小数据集、极小 batch size 的设定下，loss 波动较大是正常现象，但整体趋势仍保持缓慢下降。
3. cyberpunk_all 的后半程损失略高于 cyberpunk_portrait，说明混入 urban scene 后数据分布更复杂，学习难度也更高。

根据日志统计：

| 数据集 | 0 到 500 步平均 loss | 500 到 2000 步平均 loss | 1500 到 2000 步平均 loss |
| --- | ---: | ---: | ---: |
| mygo | 0.1121 | 0.0964 | 0.0919 |
| cyberpunk_portrait | 0.1028 | 0.0899 | 0.0889 |
| cyberpunk_all | 0.1066 | 0.1038 | 0.0977 |

这说明 PTI 训练在当前实现下是可控且可收敛的，能够支撑后续所有推理与比较实验。

## 4. 实验结果与分析

### 4.1 DreamBooth 与 PTI 的效果差异

下图比较了 Base SDXL、DreamBooth LoRA 与 PTI LoRA 在同一 mygo 半身 prompt、3 个 seed 下的结果。

![Method compare](contact_sheets/section1_method_compare.png)

观察结果如下：

1. Base SDXL 能理解“灰发、蓝眼、麦克风、校服少女”的通用语义，但角色身份不稳定。不同 seed 下会出现发型漂移和角色气质波动，本质上仍是在采样“泛化的偶像系动漫少女”。
2. DreamBooth 相较 Base SDXL 明显提升了角色相关性与校服语义，但跨 seed 的身份一致性仍不够稳定，说明它更像是把训练集风格整体注入模型，而不是学会一个强绑定的风格 token。
3. PTI 在 3 个 seed 下都更稳定地保持了短灰发、蓝眼、制服与舞台演唱语义，表现出更强的概念绑定能力。

结论：在本仓库的 SDXL 设定下，PTI 明显优于 DreamBooth，尤其适合“通过一个 token 稳定触发固定角色/风格概念”的课程展示任务。

### 4.2 四种 PTI 风格与原模型的对比

为了更全面体现最终训练成果，进一步比较了 mygo、jojo、wushan、cyberpunk 四种 PTI LoRA 与原始 SDXL 在相应 prompt 下的结果。

![Multi-style compare](contact_sheets/section1b_multi_style_compare.png)

该组实验给出几个重要现象：

1. 4 种风格全部训练成功，且都能通过各自 token 稳定触发，说明本次作业不是“只在单一风格上偶然成功”，而是已经形成了可复用的 PTI 训练流程。
2. 原始 SDXL 对 jojo、wushan、cyberpunk 这类已有强语义先验的 prompt 已经能生成“看起来还不错”的结果，这说明基座模型本身能力不弱。
3. PTI 的主要价值不在于把“完全不能生成”变成“能生成”，而在于把“泛化的类型化结果”变成“更接近特定训练集风格、更加稳定、更加可控的结果”。
4. 对 mygo 而言，PTI 更稳定地锁定了角色发型、脸型与制服样式。对 jojo 而言，PTI 显著增强了链条、帽檐、夸张阴影和漫画式五官；对 wushan 而言，PTI 更统一地保留了武侠服装、兵器与火焰动作感；对 cyberpunk 而言，PTI 更稳定地保留了白色短发、红色眼妆与特定赛博角色外观。

结论：从成果展示角度看，本次作业最有说服力的部分之一，就是 PTI 在 4 种风格上都能相对一致地优于原模型，从而证明训练并非局部成功，而是整体成功。

### 4.3 PTI 不同步数与 LoRA scale 的影响

下图给出了 mygo PTI 在 600、1000、2000 步三个 checkpoint，以及 0.4、0.8、1.2 三档 LoRA scale 下的结果对比。

![Step scale](contact_sheets/section2_step_scale.png)

主要结论如下：

1. scale 0.4 时，LoRA 影响偏弱，结果更接近基础模型，角色辨识度与风格特异性较弱。
2. scale 0.8 时，三组 checkpoint 都更稳定，其中 2000 步时综合表现最好。
3. scale 1.2 时，风格强度继续提升，但过拟合与局部失真概率明显增加。
4. 从步数看，600 步仍偏早期，1000 步已明显可用，2000 步给出最稳定的视觉质量。

综合判断：本实验中的最优平衡点仍然是 2000 步 + LoRA scale 0.8。后续大多数实验也都基于该设置展开。

### 4.4 cyberpunk 两种训练集构成的比较

下图比较了 cyberpunk_portrait 与 cyberpunk_all 两个 PTI LoRA 在 template prompt 与 wide-scene prompt 下的效果。

![Dataset compare](contact_sheets/section3_dataset_compare.png)

观察结果如下：

1. 两个模型都能较稳定保留白发、机能服、霓虹环境和赛博配色，说明 15 张 portrait 数据已经足以学习到核心人物风格。
2. portrait-only 模型在 wide-scene prompt 下仍倾向于把主体拉近，更像“角色近景 + 泛化背景”。
3. portrait + scene 的混合数据集在 wide-scene prompt 下更能保留街景、广告牌、远景透视和城市氛围。
4. 纯 portrait 版本在人像近景上略有优势，但混合版本在环境表达上显著更强。

结论：如果展示重点是角色肖像，portrait-only 已可用；如果展示重点是完整视觉世界观与场景表达，cyberpunk_all 更适合作为最终成果版本。

### 4.5 Prompt 能力与 Prompt Engineering

原始的 prompt 能力实验如下图所示。

![Prompt capability](contact_sheets/section4_prompt_capability.png)

初始观察表明：

1. mygo 的 PTI LoRA 最擅长 close-up 与 half-body。
2. full body 与 background-heavy prompt 的控制相对较弱。
3. 模型更倾向于回到它在训练数据中最熟悉的“中近景人物主导构图”。

针对这一点，进一步进行了 prompt 调优实验，对 full body 和 heavy background 提示词加入了更明确的构图约束与更合适的长宽比。

![Prompt refinement](contact_sheets/section4c_prompt_refinement.png)

调优结果给出更细致的结论：

1. full body 在 prompt refinement 后确实得到明显改善。尤其在 seed_13 与 seed_37 下，新的 prompt 更容易生成真正“头到脚可见”的完整人物，而不再只是近景假性全身像。
2. 但这一改进并非对所有 seed 都同样稳定。seed_73 仍然回退到较近的人像构图，说明 full body 能力是“可通过 prompt engineering 显著提升”，但尚未达到完全稳健。
3. heavy background 的 refinement 能在部分 seed 中增加观众、舞台与环境信息，但人物仍然往往占据画面主导位置，因此“背景主导型构图”依然是较难控制的部分。

结论：对 mygo 而言，原始模型最稳定的能力范围依然是头像与半身像；但通过更精细的 prompt 设计，full body 已经能在相当一部分样本中达到较好效果，说明其局限并非完全不可缓解。相比之下，heavy background 的提升更有限，因此仍应诚实保留这部分局限性说明。

### 4.6 四种风格之间的相互 transfer

为了展示 PTI token 并非仅能复现训练集，还能够作为“可迁移的风格控制器”，进一步设计了 4 种风格之间的 style transfer 矩阵实验。

![Style transfer matrix](contact_sheets/section4b_style_transfer_matrix.png)

该组实验展示出较强的扩展能力：

1. jojo 与 cyberpunk 的风格注入效果最强。无论目标外观来自 mygo、wushan 还是另一种风格，jojo 都能明显引入硬朗线条、夸张五官与强阴影；cyberpunk 则能稳定引入霓虹背景、冷暖对比灯光和机能装语义。
2. mygo 与 wushan 的 transfer 效果更温和，但依然清晰可辨。mygo 倾向于把人物统一成更柔和、校园动漫化的视觉风格；wushan 倾向于加入武侠服饰、动作感与火焰/水墨元素。
3. 多数组合下，模型并不是简单复制原训练图，而是在保留目标外观关键词的同时，向目标风格空间做可见迁移。这说明 PTI token 学到的不是单张图记忆，而是具有一定抽象性的风格表达能力。

结论：style transfer 是本次作业里很能体现“训练成果已经超出基础复现”的部分。特别是 jojo 与 cyberpunk token，已经表现出较强的跨概念风格控制能力，能显著增强报告的展示效果。

### 4.7 LoRA merge：2 路、3 路与 4 路的分层比较

上一轮报告只比较了单独 LoRA 与 4 路 merge。考虑到工作区中还存在 2 路 mygo+jojo 和 3 路 mygo+jojo+wushan merged 权重，本版进一步补做了分层 merge 对比。根据输出目录的实际文件，三路 merge 对应的是 mygo+jojo+wushan。

![Merge effect](contact_sheets/section5_merge_effect.png)

![Partial merge effect](contact_sheets/section5b_partial_merge_effect.png)

新的结论比上一版更细致，也更符合你对实验现象的直观观察：

1. 2 路 merge 的效果明显好于 4 路 merge。对于 mygo 与 jojo，2 路 merge 后仍然能够保持较好的角色辨识度和主要风格特征，没有出现明显的全面塌缩。
2. 3 路 merge 开始出现较明显退化，但仍然优于 4 路 merge。mygo 会出现背景与色调统一化，jojo 的专属服装和构图符号也被削弱，wushan 仍保留部分动作感与火焰元素，但精细风格已经下降。
3. 4 路 merge 的问题最严重，表现为风格串扰显著、统一化严重、专属 token 虽能触发但保真度明显下降。

因此，merge 的整体规律可以总结为：

1. 少量风格 merge 是可行的，特别是 2 路 merge 已达到“可展示、可用”的程度。
2. 随着 merge 风格数增加，风格保真度会逐步下降。
3. 当前仓库中的直接权重累加方式适合少量风格整合，不适合无约束地扩展到 4 路甚至更多风格。

这一结论比简单地说“merge 不行”更准确，也更能体现实验工作的细致程度。

## 5. 围绕课程任务的综合回答

结合全部实验，可以对课程作业作如下总结：

1. **任务完成度**：本次作业已经完整实现了数据集选择、模型训练、训练过程观察、结果对比和分析总结，且实验深度明显高于最低要求。
2. **方法选择**：在当前仓库与任务目标下，PTI 是比 DreamBooth 更优的方案，尤其适用于需要 token 可控性和角色一致性的场景。
3. **最佳训练与推理设置**：2000 步训练、0.8 的 LoRA scale 是最稳健的组合。
4. **数据集设计**：训练集构成会直接决定模型的构图偏好与环境表达能力；cyberpunk_all 的结果证明加入 scene 数据非常有价值。
5. **模型泛化能力**：训练后的 PTI LoRA 不仅能复现单一风格，还表现出可观的 style transfer 能力，说明其学习到的是具有抽象性的风格控制而非单纯记忆。
6. **Prompt 工程价值**：full body 的问题并不完全是模型失败，适当的长宽比与显式构图约束能够显著改善结果。
7. **Merge 结论**：2 路 merge 效果较好，3 路 merge 尚可，4 路 merge 明显劣化，说明 merge 的可行性与风格数量强相关。

## 6. 实验局限性

尽管最终结果整体较强，但仍有几个需要诚实保留的局限：

1. 本报告主要基于人工视觉评估，没有引入 FID、CLIP-score、LPIPS 等自动指标，因此结论偏定性。
2. DreamBooth 与 PTI 的方法对比主要基于 mygo，因为当前工作区里最完整、最直接可复用的 DreamBooth SDXL 权重集中在该风格上。
3. full body 能通过 prompt engineering 获得明显改善，但仍不是对所有 seed 都完全稳定。
4. heavy background 构图依然较难，说明训练数据与模型偏好仍更倾向人物主导的中近景构图。
5. LoRA merge 的实验只验证了当前仓库的直接合并方式，没有进一步尝试 merge 权重搜索或 merge 后再微调。

## 7. 结论

综合来看，本次课程作业已经较高质量地完成了 SDXL 上的 LoRA/ PTI 微调、推理与合并分析任务。最重要的结论不是“某一张图很好看”，而是：本仓库已经形成了一条在多种风格上可重复成功的 PTI SDXL 训练路线。通过这条路线，mygo、jojo、wushan、cyberpunk 四类风格都能稳定得到优于原始 SDXL 的结果；其中 mygo、jojo、cyberpunk 的最终展示效果尤其突出。对于课程报告而言，这意味着本项目不仅完成了任务要求，而且已经达到了“有代表性结果、有系统比较、有过程分析、有局限性反思”的完整实验标准。

若从最终成果的展示角度排序，本次实验中最值得强调的亮点是：

1. PTI 在四种风格上都训练成功，且整体优于原模型。
2. mygo 在 2000 步 + scale 0.8 下的结果稳定、可控，是最强的单模型展示案例。
3. jojo 与 cyberpunk 在 style transfer 中表现出很强的风格迁移能力，体现了训练成果的扩展价值。
4. 2 路 merge 没有出现明显质量崩坏，说明模型融合在小规模场景下是可行的。

因此，本次作业的主叙事应当是“训练成果整体较好，并且已经通过多角度实验得到验证”，而不是“仅仅跑通了训练过程”。局限性仍然存在，但它们更多地说明未来可继续优化的方向，而不是否定当前成果。

## 8. 复现实验资源

1. 实验规格文件：[report_assets/visual_report_experiments.json](../report_assets/visual_report_experiments.json)
2. 批量实验脚本：[scripts/run_sdxl_report_experiments.py](../scripts/run_sdxl_report_experiments.py)
3. 训练过程曲线：[plots/pti_training_loss_curves.png](plots/pti_training_loss_curves.png)
4. DreamBooth vs PTI 对比：[contact_sheets/section1_method_compare.png](contact_sheets/section1_method_compare.png)
5. 四风格 PTI vs 原模型：[contact_sheets/section1b_multi_style_compare.png](contact_sheets/section1b_multi_style_compare.png)
6. 步数与 scale 扫描：[contact_sheets/section2_step_scale.png](contact_sheets/section2_step_scale.png)
7. cyberpunk 训练集对比：[contact_sheets/section3_dataset_compare.png](contact_sheets/section3_dataset_compare.png)
8. 原始 prompt 能力对比：[contact_sheets/section4_prompt_capability.png](contact_sheets/section4_prompt_capability.png)
9. 风格 transfer 矩阵：[contact_sheets/section4b_style_transfer_matrix.png](contact_sheets/section4b_style_transfer_matrix.png)
10. prompt 调优对比：[contact_sheets/section4c_prompt_refinement.png](contact_sheets/section4c_prompt_refinement.png)
11. 四路 merge 对比：[contact_sheets/section5_merge_effect.png](contact_sheets/section5_merge_effect.png)
12. 两路/三路/四路 merge 对比：[contact_sheets/section5b_partial_merge_effect.png](contact_sheets/section5b_partial_merge_effect.png)
13. 方法对比元数据：[section1_method_compare/metadata.csv](section1_method_compare/metadata.csv)
14. 四风格对比元数据：[section1b_multi_style_compare/metadata.csv](section1b_multi_style_compare/metadata.csv)
15. 步数与 scale 元数据：[section2_step_scale/metadata.csv](section2_step_scale/metadata.csv)
16. 数据集对比元数据：[section3_dataset_compare/metadata.csv](section3_dataset_compare/metadata.csv)
17. 原始 prompt 能力元数据：[section4_prompt_capability/metadata.csv](section4_prompt_capability/metadata.csv)
18. 风格 transfer 元数据：[section4b_style_transfer_matrix/metadata.csv](section4b_style_transfer_matrix/metadata.csv)
19. prompt 调优元数据：[section4c_prompt_refinement/metadata.csv](section4c_prompt_refinement/metadata.csv)
20. 四路 merge 元数据：[section5_merge_effect/metadata.csv](section5_merge_effect/metadata.csv)
21. 两路/三路 merge 元数据：[section5b_partial_merge_effect/metadata.csv](section5b_partial_merge_effect/metadata.csv)
