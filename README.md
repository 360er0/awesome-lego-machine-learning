# Awesome LEGO Machine Learning
A curated list of resources dedicated to Machine Learning applications to LEGO bricks.

## Parts Classification
### Applications
* [Brickognize [2022.12]](https://brickognize.com/), [Talk from the author [2022.07]](https://www.youtube.com/watch?v=bzyG4Wf1Nkc) - Web application which recognizes any Lego part, minifigure, or set.
* [Bricksee [2022.11]](https://www.bricksee.app/) - Mobile application for organizing a Lego collection with a feature to detect parts from a photo.
* [BrickMonkey [2022.11]](https://brickmonkey.app/) - Mobile application to recognize minifig and parts.
* [RebrickNet [2022.01]](https://rebrickable.com/rebricknet/) - Web application on rebrickable.com. Currently, it detects and recognizes 300 different parts and an unspecified number of colors. Train using real videos and photos submitted by users.
* [Minifig Finder [2021.12]](https://www.minifigfinder.com/) - Web application for minifig identification. Uses Mask R-CNN for detecting individual parts (head, torso, and legs) and metric learning for classification. Currently, it looks abandoned and not working.
* [Unnamed mobile app [2021.10]](https://www.reddit.com/r/lego/comments/j7fzme/i_have_created_an_app_that_can_recognise_lego/), [Update 1 [2021.11]](https://www.reddit.com/r/lego/comments/jmsq5a/update_on_my_lego_recognition_app/), [Update 2 [2021.12]](https://www.reddit.com/r/lego/comments/k81z9g/update_about_my_lego_recognition_app_some/) - Unnamed and unreleased mobile app. Uses LDraw renders for training a classifier. The first version recognized only six different parts.
* [Brickit [2021.07]](https://brickit.app/) - Mobile application for detecting parts and suggesting models which can be built from those parts. 
* [Brickly [2021.04]](https://www.facebook.com/bricklyapp) - Unfinished and unreleased app for part detection and identification.
* [Instabrick](https://www.instabrick.org/), [Kickstarter [2019.10]](https://www.kickstarter.com/projects/piqabrick/piqabrick/description), [Review [2021.04]](https://brickset.com/article/58811/review-instabrick-part-identification-system) - Camera and web application for part identification and inventory.
* [BrickBanker [2020.12]](https://www.brickbanker.com/) - Mobile application which detects up to 2k different parts.

### Sorting Machines
* [DIY LEGO Sorting Machine Backgrounds on design choices [2023.12]](https://medium.com/@bricksortingmachine/diy-lego-sorting-machine-a4227e61221d) - The article looks into the details and individual components of our specific sorting machine, explaining the rationale behind various design decisions
* [Exploring LEGO Sorting Machines: A Survey of Designs [2023.12]](https://medium.com/@bricksortingmachine/lego-sorting-machine-overview-d390645759f9) - The article provides a comprehensive overview of the wide range of existing LEGO sorting machines.
* [BrickSortingMachine [2023.08]](https://www.youtube.com/@BrickSortingMachine), [Blog](https://medium.com/@BrickSortingMachine), [Code](https://github.com/BrickSortingMachine/BrickSortingMachine-sorter), [LEGO build instructions](https://github.com/BrickSortingMachine) - A LEGO brick sorting machine.
* [Lego Sorting Machine [2023.08]](https://www.instagram.com/lego.sorting.machine/) - Sorting machine [in progress].
* [Nexus [2023.03]](https://github.com/spencerhhubert/nexus) - Open-source sorting machine with CAD designs and code available.
* [Standard Sorter v1.0 [2023.01]](https://www.thirdarmrobotics.com/q_and_a.html) - The robotic arm that can sort based on color and/or size.
* [Big Robot [2022.06]](https://www.youtube.com/watch?v=Uj8ePOJEUdU) - Initiative to build Lego AI Open Source Sorting Machine.
* [Universal LEGO Sorting Machine [2022.06]](https://www.youtube.com/watch?v=9OO0SsRy6FE), [Description](https://www.robotminor.nl/the-lego-sorter-bsl-bricks/) - Sorting machine built for BSL Bricks store.
* [Deep Learning Lego Sorter [2021.11]](https://www.streamhead.com/3d%20printing/ai/2021/11/01/deep-learning-lego-sorting.html) - A sorting machine built from Lego with instruction and code available.
* [Lego Automatic Sorting LegoLAS 2.0 [2021.08]](https://www.youtube.com/watch?v=sCfN5LrUlKc), [Description (in German)](https://github.com/LegoAS/LegoAS), [CAD](https://cad.onshape.com/documents/987d7bcb5ba09db685ee5959/w/9b6ee89cc72c5f3be05c2815/e/2b4e90a536956ffc8c740721) - Student project in the Laboratory for Computer Science in Engineering and Computational Mathematics.
* [DISBY Sorting Machine [2021.01]](https://rbtx.com/en/solutions/disby-first-automatic-lego-bricks-sorting-system) - DISBY is the world's first automated Lego brick sorting system. It features a novel AI system that recognizes bricks based on a minimal formal description retrieved from the internet (e.g. size, weight, and description).
* [The World's First Universal LEGO Sorting Machine [2019.12]](https://www.youtube.com/watch?v=04JkdHEX3Yk), [Classifier [2019.12]](https://www.youtube.com/watch?v=-UGl0ZOCgwQ), [Dataset [2019.03]](https://medium.com/towards-data-science/how-i-created-over-100-000-labeled-lego-training-images-ec74191bb4ef), [CV Pipeline [2019.08]](https://towardsdatascience.com/a-high-speed-computer-vision-pipeline-for-the-universal-lego-sorting-machine-253f5a690ef4) - Sorting machine built by Daniel West.
* [The Shape Sifter [2019.06]](https://github.com/Spongeloaf/the-shape-sifter), [Blog](https://mt_pages.silvrback.com/) - The Shape Sifter is a Lego sorting machine utilizing a neural network, image processing software, a conveyor belt, and air jets. 
* [Lego Sorter using TensorFlow on Raspberry Pi [2018.09]](https://medium.com/@pacogarcia3/tensorflow-on-raspbery-pi-lego-sorter-ab60019dcf32) - Sorting machine that recognizes 11 different parts.
* [Letzgo Sorter [2018.01]](https://www.youtube.com/watch?v=Evo4AtPlvPM) - Sorting machine built for Letzgo company. There are a few more videos available but not many details.
* [Automatic Lego Sorting Machine](https://www.ceias.nau.edu/capstone/projects/ME/2019/19S1_LegoB/Final%20Report%201.pdf) - Students' report containing the design of the sorting machine. Mostly related to the mechanical side of the machine.
* [Sorting 2 Metric Tons of Lego [2017.04]](https://jacquesmattheij.com/sorting-two-metric-tons-of-lego/), [Software Side [2017.05]](https://jacquesmattheij.com/sorting-lego-the-software-side/) - One of the first publically described sorter that uses ML for part classification. 

### Other projects
* [JetClean [2022.10]](https://developer.nvidia.com/embedded/community/jetson-projects/jetclean) - JetClean is a small robotic Lego cleaner capable of autonomously navigating around your bedroom and keeping it tidy!

### Code
* [LegoSorter [2023.09]](https://github.com/LegoSorter) - Code for a mobile app that can recognize and count Lego bricks. It includes scripts for rendering the dataset, training the model, and the backend for the app.
* [OpenBlok [2022.11]](https://github.com/blokbot-io/OpenBlok) - OpenBlok is an open-source Lego identification and sorting system using AI models developed by blokbot.io
* [Lego Brick Recognition [2020.03]](https://github.com/jtheiner/LegoBrickClassification) - Code for generating synthetic dataset and training a classifier for 15 different parts.
* [Lego Classifier [2019.08]](https://ladvien.com/lego-deep-learning-classifier/) - Detailed description of training a part classifier and deploying it on Arduino.
* [Lego Detector [2019.03]](https://github.com/kirill-sidorchuk/lego_detector) - Code for training a classifer.

### Papers
* [Brickinspector [2023.02]](https://www.tramacsoft.com/brickinspector/), [Paper](https://www.mdpi.com/1424-8220/23/4/1898) - The paper describes the process of using synthetic data for training semantic segmentation model for detecting Lego parts.
* [How to sort them? A network for LEGO bricks classification [2022.07]](https://www.iccs-meeting.org/archive/iccs2022/papers/133520608.pdf) - The paper presents a comparison of 28 models used for image classification trained to recognize 447 different LEGO bricks.
* [Hierarchical 2-step neural-based LEGO bricks detection and labeling [2021.04]](https://mostwiedzy.pl/en/publication/hierarchical-2-step-neural-based-lego-bricks-detection-and-labeling,155119-1) - The paper proposes two-step system for identifying LEGO bricks -- detection and classification. The model is limited to recognizing only 10 different parts.
* [Lego Recognition Tool [2018.03]](https://robo4you.at/publications/Lego.pdf) - The diploma thesis describes the process of developing a machine, that is capable of identifying Lego bricks and sorting them into boxes.

### Datasets
* [B200 LEGO Detection Dataset [2024.03]](https://www.kaggle.com/datasets/ronanpickell/b100-lego-detection-dataset) - Dataset for parts detection. It contains 2k high-quality renders for 200 different parts.
* [Photos and rendered images of LEGO bricks [2023.11]](https://www.nature.com/articles/s41597-023-02682-2) - The paper describes a collection of datasets containing both LEGO brick renders and real photos. The datasets contain around 155,000 photos and nearly 1,500,000 renders.
* [Video of LEGO bricks on conveyor belt [2022.01]](https://mostwiedzy.pl/en/open-research-data-series/video-of-lego-bricks-on-conveyor-belt,202011132226557715481-0/catalog) - The dataset contains videos of LEGO bricks moving on a white conveyor belt to train a classifier for sorting machine.
* [B200C LEGO Classification Dataset [2021.08]](https://www.kaggle.com/datasets/ronanpickell/b200c-lego-classification-dataset), [Code](https://github.com/korra-pickell/LEGO-Classification-Dataset) - Dataset for parts classification. It contains 800k high-quality renders for 200 different parts.
* [LEGO bricks for training classification network [2021.06]](https://mostwiedzy.pl/en/open-research-data/lego-bricks-for-training-classification-network,618104539639776-0) - The dataset part classification. It contains images of 447 different parts, both real photos (52k) and renders (567k). 
* [Tagged images with LEGO bricks [2021.02]](https://mostwiedzy.pl/en/open-research-data/tagged-images-with-lego-bricks,209111650250426-0) - The dataset for parts detection. It contains 2933 photos and 2908 renders annotated with bounding boxes. It doesn't include information about part IDs.
* [Lego Brick Sorting [2018.12]](https://www.kaggle.com/datasets/pacogarciam3/lego-brick-sorting-image-recognition) - The dataset for parts classification. It contains 4,580 photos of 20 different parts.
* [Lego vs. Generic Brick [2018.12]](https://www.kaggle.com/datasets/pacogarciam3/lego-vs-generic-brick-image-recognition) - The dataset for recognizing original vs fake bricks. It contains 12 classes across 6 brick types, and more than 20k images taken by 4 cameras.

### Rendering Parts
* [Lego Rendering Pipeline [2023.06]](https://github.com/brianlow/lego-rendering) - Rendering pipeline for semi-realistic, individual parts.
* [Lego multi object detection [2021.11]](https://github.com/mantyni/Multi-object-detection-lego) - Script to generate renders of LEGO parts and corresponding bounding boxes. Uses Python and Blender.
* [BrickRegistration [2021.11]](https://github.com/GistNoesis/BrickRegistration) - A tool to generate synthetic 3d scenes with LEGO parts and their segmentation information.
* [Lego Renderer for ML Projects [2020.01]](https://github.com/WHSnyder/LegoTrainingRenderer) - A set of Python scripts/Blender utilities for rendering Lego scenes for use in deep learning/computer vision projects. Includes a basic scene with a tracked camera, scripts for rendering images, normals, masks of Lego combinations, and utilities for recording the positions of special features on different pieces (studs, corners, holes) easily.
* [Rendering LDraw Parts Images for Rebrickable [2018.10]](https://bricksafe.com/files/Simon/guide/guide.html) - Comprehensive guide describing how Rebrickable rendered their images.

### Understanding Part IDs and colors
* [Rebrickable numbering [2023.02]](https://rebrickable.com/help/part-numbering/)
* [Understanding LEGO part numbers [2020.10]](https://brickset.com/article/54327/understanding-lego-part-numbers)
* [The curious case of LEGO colors [2016.09]](https://www.bartneck.de/2016/09/09/the-curious-case-of-lego-colors/) - This post explores the difficulty of defining the exact LEGO color palette. It discusses the various lists, conversions, and color systems.
* [Let's talk about classification of parts](https://swooshable.com/parts/classification)
* [LDraw Part Number Specification](https://www.ldraw.org/part-number-spec.html)
* [Bricklink Item Numbers](https://www.bricklink.com/help.asp?helpID=168)


## Generating Sets
### Papers
* [Brick Yourself within 3 Minutes [2022.05]](https://air.tsinghua.edu.cn/en/Brick-Yourself-within-3-Minutes.pdf) - Given the input portrait, it uses several deep neural networks to extract the attributes to describe the human appearance shown in the image. Built on these attributes, the model generates the corresponding uncolored brick model by iteratively searching for brick components with the coordinate descent algorithm. Finally, it assigns a color to every brick to get the final brick model.
* [Image2Lego: Customized LEGO® Set Generation from Images [2021.08]](https://arxiv.org/abs/2108.08477) - The system takes an image as an input, encodes it as embedding, generates a voxelized 3D model, and finally convert it into LEGO bricks. The model jointly trains a 3D model autoencoder and a 2D-to-3D encoder.
* [Automatic Generation of Vivid LEGO Architectural Sculptures [2019]](http://staff.ustc.edu.cn/~xjchen99/2019cgf-lego-zhou.pdf) - It introduces an automatic system to convert an architectural model into a LEGO sculpture while preserving the original model’s shape features like repeating components, shape details and planarity.
* [Legolization: Optimizing LEGO Designs [2015]](http://www.cs.columbia.edu/~yonghao/siga15/luo-Legolization.pdf) - It converts a 3D model into a voxelized 3D model, and then into LEGO bricks. The paper focuses on optimizing the physical stability of the generated design.
* [Survey on Automated LEGO Assembly Construction [2014]](https://core.ac.uk/download/pdf/295560271.pdf) - The survey paper on converting 3D models into LEGO models. 

### Datasets
* [Lego sets made in LDraw](https://www.eurobricks.com/forum/index.php?/forums/topic/48285-key-topic-official-lego-sets-made-in-ldraw/) - Links to over 2000 LEGO sets recreated in LDraw.
* [LDraw Official Model Repository](https://omr.ldraw.org/) - LDraw repository of over 1800 official LEGO sets. 
* [BrickHub](https://brickhub.org/) - Almost 500 original and custom sets.

## Assembling Sets
### Papers
* [Planning Assembly Sequence with Graph Transformer [2022.10]](https://arxiv.org/abs/2210.05236) - A graph transformer-based framework for ASP problem, with a heterogeneous graph attention network to encoder the models, which are decoded with the attention mechanism to generate assembly sequence.
* [Translating a Visual LEGO Manual to a Machine-Executable Plan [2022.07]](https://cs.stanford.edu/~rcwang/projects/lego_manual/) - Understanding the assembly process using manual.
* [Break and Make: Interactive Structural Understanding Using LEGO Bricks [2022.07]](https://arxiv.org/abs/2207.13738) - Introduction of a new task for visual understanding, 3D simulator to manipulate LEGO models, and a model to solve the proposed task of recreating the LEGO model.
* [Building LEGO Using Deep Generative Models of Graphs [2020.12]](https://arxiv.org/abs/2012.11543) - It proposes a way to represent the LEGO model as graphs and learn how to generate them step-by-step.

## Generating images of LEGO
### Models
* [Lego Minifig XL [2023.08]](https://huggingface.co/nerijs/lego-minifig-xl) - Stable Diffusion model for generating Minifigures.
* [Lego Brickheadz XL [2023.09]](https://huggingface.co/nerijs/lego-brickheadz-xl) - Stable Diffusion model for generating Brickheadz.

### Posts
* [The AI Revolution: How Artificial Intelligence Is Impacting the LEGO Community [2023.11]](https://bricknerd.com/home/the-ai-revolution-how-artificial-intelligence-is-impacting-the-lego-community-11-7-23) - The post discusses how AI-generated “LEGO” sets are sparking a debate between inspiration and imitation within the AFOL community.
* [Reimagining LEGO sets [2023.03]](https://brickset.com/article/92515/reimagining-lego-sets) - Generating realistic versions of LEGO sets.
* [LEGO Stable Diffusion [2023.01]](https://github.com/MichWozPol/LEGO_StableDiffusion) - Fine-tuned stable diffusion model for generating images in the LEGO style.
* [Generating LEGO Pirates sets and minifigures [2023.01]](https://www.eurobricks.com/forum/index.php?/forums/topic/193551-i-fed-an-ai-image-generator-with-lego-pirate-prompts-and-this-is-what-happened-lots-of-images/) - Generating LEGO Pirates sets and minifigures using Stable Diffusion.
* [Using AI to generate minifigures [2020.07]](https://brickset.com/article/52483/using-ai-to-generate-minifigures), [Part 2 [2020.08]](https://brickset.com/article/53051/using-ai-to-generate-minifigures-part-2), [Part 3 [2020.08]](https://brickset.com/article/63365/using-ai-to-generate-minifigures-part-3) - Using different GANs to generate images of minifigures.

## ML at LEGO Group
* [Building a GenAI Solution — A Hackathon Success Story! [2023.11]](https://medium.com/lego-engineering/a-hackathon-success-story-using-generative-ai-f99ae4f09d88) - Description of the winning project of the internal Generative AI Hackathon at the LEGO Group. It focuses on generating a set description based on metadata and an image.
* [AI Summit London: Lego’s Brian Schwab on Interaction Design [2023.06]](https://www.youtube.com/watch?v=DaactYVgEVQ) - A short talk with Brian Schwab without many details about the augmented reality for playing with Lego bricks.
* [A One-Stop Data Shop: The Lego Group’s Anders Butzbach Christensen [2023.03]](https://sloanreview.mit.edu/audio/a-one-stop-data-shop-the-lego-groups-anders-butzbach-christensen/) - Podcast (and transcript) about LEGO data platform.
* [How LEGO plays with data: An interview with chief data officer Orlando Machado [2022.12]](https://www.mckinsey.com/capabilities/quantumblack/our-insights/how-lego-plays-with-data-an-interview-with-chief-data-officer-orlando-machado)
* [Inside the annual LEGO Brick Hack [2022.11]](https://www.lego.com/en-gb/careers/stories/inside-the-annual-lego-brick-hack) - Short description of internal Data Science hackathon.
* [Building Blocks of Machine Learning at LEGO with Francesc Joan Riera [2021.11]](https://twimlai.com/podcast/twimlai/building-blocks-machine-learning-lego-francesc-joan-riera/) - Podcast about using ML at LEGO. In particular, their ML infrastructure, content moderation system, and classifying images of LEGO models into thematical categories.

## Other lists
* [Awesome LEGO](https://github.com/ad-si/awesome-lego) - General list of LEGO resources.
