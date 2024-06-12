# Scene-Generator
### Abstract
Generative AI is in the preliminary stages of supporting 3D content creation. This projects aims to use existing generative models in conjunction with post-processing to generate realistic 3D scenes to be used in production software. The implementation of the scene generation is in the form of a tool independent of content creation software. The output is in USD format which is easily imported into the workflow.
### Steps
1. Install dependencies.
2. Install OpenUSD and add libraries to environment variables. OS-specific instructions can be found [here](https://github.com/PixarAnimationStudios/OpenUSD).
3. If using models of non-USD file format, we use [this](https://github.com/adobe/USD-Fileformat-plugins) plugin by Adobe to convert model files used in the scene. You may use the scene generation script directly is model files are already in usd format. Make sure model files are named exactly as they are detected in the detection stage.
4. Use usdview to view the scene within the built-in OpenUSD renderer. Alternatively, and perhaps this is the intended use case, import the USD file in a content creation software of choice. Most software suites are supported but you may refer to native documentation in case there are other conditions.
