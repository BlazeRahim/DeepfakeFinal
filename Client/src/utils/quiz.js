const quiz=[
    {
        edition:0,
        name:"celebrity edition",
        heading:"Test your skills and see if you can spot the difference between real videos and deepfake creations ",
        context:"without audio",
        focus:"Focus on face masking and lip syncing to spot the difference between real and fake.",
        cover: "https://firebasestorage.googleapis.com/v0/b/veriface-19211.appspot.com/o/images%2FGroup%205.png?alt=media&token=f5c025bf-003e-4080-8543-79c91c0ca283&_gl=1*w03kcy*_ga*MTcxNDE5NzgzLjE2ODcwNzU0MTQ.*_ga_CW55HF8NVT*MTY5OTM1Mjc1Ni4yNC4xLjE2OTkzNTI4MDQuMTIuMC4w",
        data:[
            {
                video:"https://firebasestorage.googleapis.com/v0/b/veriface-19211.appspot.com/o/video%2Ffake1.mp4?alt=media&token=b08e151e-64c8-4ad8-88d2-cc1abe1c4810&_gl=1*gx1cun*_ga*MTcxNDE5NzgzLjE2ODcwNzU0MTQ.*_ga_CW55HF8NVT*MTY5OTE5NzQxOS4yMy4xLjE2OTkxOTc2MDcuNTkuMC4w",
                real:"https://firebasestorage.googleapis.com/v0/b/veriface-19211.appspot.com/o/video%2Freal1.mp4?alt=media&token=1bf2691e-55fe-4fb0-ba45-3acdc673244a&_gl=1*1vze2f6*_ga*MTcxNDE5NzgzLjE2ODcwNzU0MTQ.*_ga_CW55HF8NVT*MTY5OTE5NzQxOS4yMy4xLjE2OTkxOTc1MzguNjAuMC4w",
                result:0
            },
            {
                video:"https://firebasestorage.googleapis.com/v0/b/veriface-19211.appspot.com/o/video%2Freal2.mp4?alt=media&token=d59c9443-c530-4b0e-ac77-00e00c44b332&_gl=1*13f0p1y*_ga*MTcxNDE5NzgzLjE2ODcwNzU0MTQ.*_ga_CW55HF8NVT*MTY5OTE5NzQxOS4yMy4xLjE2OTkxOTc2MzYuMzAuMC4w" ,
                result:1
            },
            {
                video:"https://firebasestorage.googleapis.com/v0/b/veriface-19211.appspot.com/o/video%2Ffake3.mp4?alt=media&token=d5bb9f4b-c14e-40cb-acd6-a19c097d483c&_gl=1*w06r9c*_ga*MTcxNDE5NzgzLjE2ODcwNzU0MTQ.*_ga_CW55HF8NVT*MTY5OTE5NzQxOS4yMy4xLjE2OTkxOTc3MTguMzcuMC4w" ,
                real:"https://firebasestorage.googleapis.com/v0/b/veriface-19211.appspot.com/o/video%2Freal3.mp4?alt=media&token=602c4335-7589-407e-bec1-de4206eb0751&_gl=1*9b06ap*_ga*MTcxNDE5NzgzLjE2ODcwNzU0MTQ.*_ga_CW55HF8NVT*MTY5OTE5NzQxOS4yMy4xLjE2OTkxOTc2OTUuNjAuMC4w",
                result:0
            },
            {
                video: "https://firebasestorage.googleapis.com/v0/b/veriface-19211.appspot.com/o/video%2Ffake4.mp4?alt=media&token=576070db-a413-4c22-abf8-b8baa2fd4e8c&_gl=1*1all0f4*_ga*MTcxNDE5NzgzLjE2ODcwNzU0MTQ.*_ga_CW55HF8NVT*MTY5OTE5NzQxOS4yMy4xLjE2OTkxOTc3NTguNjAuMC4w",
                real:"https://firebasestorage.googleapis.com/v0/b/veriface-19211.appspot.com/o/video%2Freal4.mp4?alt=media&token=10fbaa14-a2cc-4a52-afe2-358885f8b8a1&_gl=1*ffznsh*_ga*MTcxNDE5NzgzLjE2ODcwNzU0MTQ.*_ga_CW55HF8NVT*MTY5OTE5NzQxOS4yMy4xLjE2OTkxOTc3NDEuMTQuMC4w" ,
                result:0
            },
            {
                video:"https://firebasestorage.googleapis.com/v0/b/veriface-19211.appspot.com/o/video%2Freal5.mp4?alt=media&token=e691b29c-90b3-4943-ab49-769dc3415135&_gl=1*112w89h*_ga*MTcxNDE5NzgzLjE2ODcwNzU0MTQ.*_ga_CW55HF8NVT*MTY5OTE5NzQxOS4yMy4xLjE2OTkxOTc3ODYuMzIuMC4w",
                result:1
            },
            {
                video:"https://firebasestorage.googleapis.com/v0/b/veriface-19211.appspot.com/o/video%2Ffake6.mp4?alt=media&token=dd29e0fa-c36a-4e36-8548-002c4b7e4874&_gl=1*11xxqnc*_ga*MTcxNDE5NzgzLjE2ODcwNzU0MTQ.*_ga_CW55HF8NVT*MTY5OTE5NzQxOS4yMy4xLjE2OTkxOTc4MjQuNTkuMC4w",
                real:"https://firebasestorage.googleapis.com/v0/b/veriface-19211.appspot.com/o/video%2Freal6.mp4?alt=media&token=6bf4ea21-9235-482a-8ec1-dcfe88667761&_gl=1*d7brnd*_ga*MTcxNDE5NzgzLjE2ODcwNzU0MTQ.*_ga_CW55HF8NVT*MTY5OTE5NzQxOS4yMy4xLjE2OTkxOTc4MzIuNTEuMC4w",
                result:0
            }
        ]
    },
    {
        edition:1,
        name:"Politician Edition",
        heading:"Test your skills and see if you can spot the difference between real videos and deepfake creations ",
        context:"with audio",
        focus:"Focus on face masking and lip syncing and audio to spot the difference between real and fake.",
        cover:"https://firebasestorage.googleapis.com/v0/b/veriface-19211.appspot.com/o/images%2FGroup%203.png?alt=media&token=19adeebd-a9d6-40e6-ab92-3cc1f6ee5ec4&_gl=1*i1nt5h*_ga*MTcxNDE5NzgzLjE2ODcwNzU0MTQ.*_ga_CW55HF8NVT*MTY5OTM1Mjc1Ni4yNC4xLjE2OTkzNTM0NDYuNTYuMC4w",
        data:[
            {
                video:"https://firebasestorage.googleapis.com/v0/b/veriface-19211.appspot.com/o/video%2Ffake1.mp4?alt=media&token=b08e151e-64c8-4ad8-88d2-cc1abe1c4810&_gl=1*gx1cun*_ga*MTcxNDE5NzgzLjE2ODcwNzU0MTQ.*_ga_CW55HF8NVT*MTY5OTE5NzQxOS4yMy4xLjE2OTkxOTc2MDcuNTkuMC4w",
                real:"https://firebasestorage.googleapis.com/v0/b/veriface-19211.appspot.com/o/video%2Freal1.mp4?alt=media&token=1bf2691e-55fe-4fb0-ba45-3acdc673244a&_gl=1*1vze2f6*_ga*MTcxNDE5NzgzLjE2ODcwNzU0MTQ.*_ga_CW55HF8NVT*MTY5OTE5NzQxOS4yMy4xLjE2OTkxOTc1MzguNjAuMC4w",
                result:0
            },
            {
                video:"https://firebasestorage.googleapis.com/v0/b/veriface-19211.appspot.com/o/video%2Freal2.mp4?alt=media&token=d59c9443-c530-4b0e-ac77-00e00c44b332&_gl=1*13f0p1y*_ga*MTcxNDE5NzgzLjE2ODcwNzU0MTQ.*_ga_CW55HF8NVT*MTY5OTE5NzQxOS4yMy4xLjE2OTkxOTc2MzYuMzAuMC4w" ,
                result:1
            },
            {
                video:"https://firebasestorage.googleapis.com/v0/b/veriface-19211.appspot.com/o/video%2Ffake3.mp4?alt=media&token=d5bb9f4b-c14e-40cb-acd6-a19c097d483c&_gl=1*w06r9c*_ga*MTcxNDE5NzgzLjE2ODcwNzU0MTQ.*_ga_CW55HF8NVT*MTY5OTE5NzQxOS4yMy4xLjE2OTkxOTc3MTguMzcuMC4w" ,
                real:"https://firebasestorage.googleapis.com/v0/b/veriface-19211.appspot.com/o/video%2Freal3.mp4?alt=media&token=602c4335-7589-407e-bec1-de4206eb0751&_gl=1*9b06ap*_ga*MTcxNDE5NzgzLjE2ODcwNzU0MTQ.*_ga_CW55HF8NVT*MTY5OTE5NzQxOS4yMy4xLjE2OTkxOTc2OTUuNjAuMC4w",
                result:0
            },
            {
                video: "https://firebasestorage.googleapis.com/v0/b/veriface-19211.appspot.com/o/video%2Ffake4.mp4?alt=media&token=576070db-a413-4c22-abf8-b8baa2fd4e8c&_gl=1*1all0f4*_ga*MTcxNDE5NzgzLjE2ODcwNzU0MTQ.*_ga_CW55HF8NVT*MTY5OTE5NzQxOS4yMy4xLjE2OTkxOTc3NTguNjAuMC4w",
                real:"https://firebasestorage.googleapis.com/v0/b/veriface-19211.appspot.com/o/video%2Freal4.mp4?alt=media&token=10fbaa14-a2cc-4a52-afe2-358885f8b8a1&_gl=1*ffznsh*_ga*MTcxNDE5NzgzLjE2ODcwNzU0MTQ.*_ga_CW55HF8NVT*MTY5OTE5NzQxOS4yMy4xLjE2OTkxOTc3NDEuMTQuMC4w" ,
                result:0
            },
            {
                video:"https://firebasestorage.googleapis.com/v0/b/veriface-19211.appspot.com/o/video%2Freal5.mp4?alt=media&token=e691b29c-90b3-4943-ab49-769dc3415135&_gl=1*112w89h*_ga*MTcxNDE5NzgzLjE2ODcwNzU0MTQ.*_ga_CW55HF8NVT*MTY5OTE5NzQxOS4yMy4xLjE2OTkxOTc3ODYuMzIuMC4w",
                result:1
            },
            {
                video:"https://firebasestorage.googleapis.com/v0/b/veriface-19211.appspot.com/o/video%2Ffake6.mp4?alt=media&token=dd29e0fa-c36a-4e36-8548-002c4b7e4874&_gl=1*11xxqnc*_ga*MTcxNDE5NzgzLjE2ODcwNzU0MTQ.*_ga_CW55HF8NVT*MTY5OTE5NzQxOS4yMy4xLjE2OTkxOTc4MjQuNTkuMC4w",
                real:"https://firebasestorage.googleapis.com/v0/b/veriface-19211.appspot.com/o/video%2Freal6.mp4?alt=media&token=6bf4ea21-9235-482a-8ec1-dcfe88667761&_gl=1*d7brnd*_ga*MTcxNDE5NzgzLjE2ODcwNzU0MTQ.*_ga_CW55HF8NVT*MTY5OTE5NzQxOS4yMy4xLjE2OTkxOTc4MzIuNTEuMC4w",
                result:0
            }
        ]
    } , 
    {
        edition:0,
        name:"celebrity edition",
        heading:"Test your skills and see if you can spot the difference between real videos and deepfake creations ",
        context:"without audio",
        focus:"Focus on face masking and lip syncing to spot the difference between real and fake.",
        cover: "https://firebasestorage.googleapis.com/v0/b/veriface-19211.appspot.com/o/images%2FGroup%205.png?alt=media&token=f5c025bf-003e-4080-8543-79c91c0ca283&_gl=1*w03kcy*_ga*MTcxNDE5NzgzLjE2ODcwNzU0MTQ.*_ga_CW55HF8NVT*MTY5OTM1Mjc1Ni4yNC4xLjE2OTkzNTI4MDQuMTIuMC4w",
        data:[
            {
                video:"https://firebasestorage.googleapis.com/v0/b/veriface-19211.appspot.com/o/video%2Ffake1.mp4?alt=media&token=b08e151e-64c8-4ad8-88d2-cc1abe1c4810&_gl=1*gx1cun*_ga*MTcxNDE5NzgzLjE2ODcwNzU0MTQ.*_ga_CW55HF8NVT*MTY5OTE5NzQxOS4yMy4xLjE2OTkxOTc2MDcuNTkuMC4w",
                real:"https://firebasestorage.googleapis.com/v0/b/veriface-19211.appspot.com/o/video%2Freal1.mp4?alt=media&token=1bf2691e-55fe-4fb0-ba45-3acdc673244a&_gl=1*1vze2f6*_ga*MTcxNDE5NzgzLjE2ODcwNzU0MTQ.*_ga_CW55HF8NVT*MTY5OTE5NzQxOS4yMy4xLjE2OTkxOTc1MzguNjAuMC4w",
                result:0
            },
            {
                video:"https://firebasestorage.googleapis.com/v0/b/veriface-19211.appspot.com/o/video%2Freal2.mp4?alt=media&token=d59c9443-c530-4b0e-ac77-00e00c44b332&_gl=1*13f0p1y*_ga*MTcxNDE5NzgzLjE2ODcwNzU0MTQ.*_ga_CW55HF8NVT*MTY5OTE5NzQxOS4yMy4xLjE2OTkxOTc2MzYuMzAuMC4w" ,
                result:1
            },
            {
                video:"https://firebasestorage.googleapis.com/v0/b/veriface-19211.appspot.com/o/video%2Ffake3.mp4?alt=media&token=d5bb9f4b-c14e-40cb-acd6-a19c097d483c&_gl=1*w06r9c*_ga*MTcxNDE5NzgzLjE2ODcwNzU0MTQ.*_ga_CW55HF8NVT*MTY5OTE5NzQxOS4yMy4xLjE2OTkxOTc3MTguMzcuMC4w" ,
                real:"https://firebasestorage.googleapis.com/v0/b/veriface-19211.appspot.com/o/video%2Freal3.mp4?alt=media&token=602c4335-7589-407e-bec1-de4206eb0751&_gl=1*9b06ap*_ga*MTcxNDE5NzgzLjE2ODcwNzU0MTQ.*_ga_CW55HF8NVT*MTY5OTE5NzQxOS4yMy4xLjE2OTkxOTc2OTUuNjAuMC4w",
                result:0
            },
            {
                video: "https://firebasestorage.googleapis.com/v0/b/veriface-19211.appspot.com/o/video%2Ffake4.mp4?alt=media&token=576070db-a413-4c22-abf8-b8baa2fd4e8c&_gl=1*1all0f4*_ga*MTcxNDE5NzgzLjE2ODcwNzU0MTQ.*_ga_CW55HF8NVT*MTY5OTE5NzQxOS4yMy4xLjE2OTkxOTc3NTguNjAuMC4w",
                real:"https://firebasestorage.googleapis.com/v0/b/veriface-19211.appspot.com/o/video%2Freal4.mp4?alt=media&token=10fbaa14-a2cc-4a52-afe2-358885f8b8a1&_gl=1*ffznsh*_ga*MTcxNDE5NzgzLjE2ODcwNzU0MTQ.*_ga_CW55HF8NVT*MTY5OTE5NzQxOS4yMy4xLjE2OTkxOTc3NDEuMTQuMC4w" ,
                result:0
            },
            {
                video:"https://firebasestorage.googleapis.com/v0/b/veriface-19211.appspot.com/o/video%2Freal5.mp4?alt=media&token=e691b29c-90b3-4943-ab49-769dc3415135&_gl=1*112w89h*_ga*MTcxNDE5NzgzLjE2ODcwNzU0MTQ.*_ga_CW55HF8NVT*MTY5OTE5NzQxOS4yMy4xLjE2OTkxOTc3ODYuMzIuMC4w",
                result:1
            },
            {
                video:"https://firebasestorage.googleapis.com/v0/b/veriface-19211.appspot.com/o/video%2Ffake6.mp4?alt=media&token=dd29e0fa-c36a-4e36-8548-002c4b7e4874&_gl=1*11xxqnc*_ga*MTcxNDE5NzgzLjE2ODcwNzU0MTQ.*_ga_CW55HF8NVT*MTY5OTE5NzQxOS4yMy4xLjE2OTkxOTc4MjQuNTkuMC4w",
                real:"https://firebasestorage.googleapis.com/v0/b/veriface-19211.appspot.com/o/video%2Freal6.mp4?alt=media&token=6bf4ea21-9235-482a-8ec1-dcfe88667761&_gl=1*d7brnd*_ga*MTcxNDE5NzgzLjE2ODcwNzU0MTQ.*_ga_CW55HF8NVT*MTY5OTE5NzQxOS4yMy4xLjE2OTkxOTc4MzIuNTEuMC4w",
                result:0
            }
        ]
    },
    {
        edition:1,
        name:"Politician edition",
        heading:"Test your skills and see if you can spot the difference between real videos and deepfake creations ",
        context:"with audio",
        focus:"Focus on face masking and lip syncing and audio to spot the difference between real and fake.",
        cover:"https://firebasestorage.googleapis.com/v0/b/veriface-19211.appspot.com/o/images%2FGroup%203.png?alt=media&token=19adeebd-a9d6-40e6-ab92-3cc1f6ee5ec4&_gl=1*i1nt5h*_ga*MTcxNDE5NzgzLjE2ODcwNzU0MTQ.*_ga_CW55HF8NVT*MTY5OTM1Mjc1Ni4yNC4xLjE2OTkzNTM0NDYuNTYuMC4w",
        data:[
            {
                video:"https://firebasestorage.googleapis.com/v0/b/veriface-19211.appspot.com/o/video%2Ffake1.mp4?alt=media&token=b08e151e-64c8-4ad8-88d2-cc1abe1c4810&_gl=1*gx1cun*_ga*MTcxNDE5NzgzLjE2ODcwNzU0MTQ.*_ga_CW55HF8NVT*MTY5OTE5NzQxOS4yMy4xLjE2OTkxOTc2MDcuNTkuMC4w",
                real:"https://firebasestorage.googleapis.com/v0/b/veriface-19211.appspot.com/o/video%2Freal1.mp4?alt=media&token=1bf2691e-55fe-4fb0-ba45-3acdc673244a&_gl=1*1vze2f6*_ga*MTcxNDE5NzgzLjE2ODcwNzU0MTQ.*_ga_CW55HF8NVT*MTY5OTE5NzQxOS4yMy4xLjE2OTkxOTc1MzguNjAuMC4w",
                result:0
            },
            {
                video:"https://firebasestorage.googleapis.com/v0/b/veriface-19211.appspot.com/o/video%2Freal2.mp4?alt=media&token=d59c9443-c530-4b0e-ac77-00e00c44b332&_gl=1*13f0p1y*_ga*MTcxNDE5NzgzLjE2ODcwNzU0MTQ.*_ga_CW55HF8NVT*MTY5OTE5NzQxOS4yMy4xLjE2OTkxOTc2MzYuMzAuMC4w" ,
                result:1
            },
            {
                video:"https://firebasestorage.googleapis.com/v0/b/veriface-19211.appspot.com/o/video%2Ffake3.mp4?alt=media&token=d5bb9f4b-c14e-40cb-acd6-a19c097d483c&_gl=1*w06r9c*_ga*MTcxNDE5NzgzLjE2ODcwNzU0MTQ.*_ga_CW55HF8NVT*MTY5OTE5NzQxOS4yMy4xLjE2OTkxOTc3MTguMzcuMC4w" ,
                real:"https://firebasestorage.googleapis.com/v0/b/veriface-19211.appspot.com/o/video%2Freal3.mp4?alt=media&token=602c4335-7589-407e-bec1-de4206eb0751&_gl=1*9b06ap*_ga*MTcxNDE5NzgzLjE2ODcwNzU0MTQ.*_ga_CW55HF8NVT*MTY5OTE5NzQxOS4yMy4xLjE2OTkxOTc2OTUuNjAuMC4w",
                result:0
            },
            {
                video: "https://firebasestorage.googleapis.com/v0/b/veriface-19211.appspot.com/o/video%2Ffake4.mp4?alt=media&token=576070db-a413-4c22-abf8-b8baa2fd4e8c&_gl=1*1all0f4*_ga*MTcxNDE5NzgzLjE2ODcwNzU0MTQ.*_ga_CW55HF8NVT*MTY5OTE5NzQxOS4yMy4xLjE2OTkxOTc3NTguNjAuMC4w",
                real:"https://firebasestorage.googleapis.com/v0/b/veriface-19211.appspot.com/o/video%2Freal4.mp4?alt=media&token=10fbaa14-a2cc-4a52-afe2-358885f8b8a1&_gl=1*ffznsh*_ga*MTcxNDE5NzgzLjE2ODcwNzU0MTQ.*_ga_CW55HF8NVT*MTY5OTE5NzQxOS4yMy4xLjE2OTkxOTc3NDEuMTQuMC4w" ,
                result:0
            },
            {
                video:"https://firebasestorage.googleapis.com/v0/b/veriface-19211.appspot.com/o/video%2Freal5.mp4?alt=media&token=e691b29c-90b3-4943-ab49-769dc3415135&_gl=1*112w89h*_ga*MTcxNDE5NzgzLjE2ODcwNzU0MTQ.*_ga_CW55HF8NVT*MTY5OTE5NzQxOS4yMy4xLjE2OTkxOTc3ODYuMzIuMC4w",
                result:1
            },
            {
                video:"https://firebasestorage.googleapis.com/v0/b/veriface-19211.appspot.com/o/video%2Ffake6.mp4?alt=media&token=dd29e0fa-c36a-4e36-8548-002c4b7e4874&_gl=1*11xxqnc*_ga*MTcxNDE5NzgzLjE2ODcwNzU0MTQ.*_ga_CW55HF8NVT*MTY5OTE5NzQxOS4yMy4xLjE2OTkxOTc4MjQuNTkuMC4w",
                real:"https://firebasestorage.googleapis.com/v0/b/veriface-19211.appspot.com/o/video%2Freal6.mp4?alt=media&token=6bf4ea21-9235-482a-8ec1-dcfe88667761&_gl=1*d7brnd*_ga*MTcxNDE5NzgzLjE2ODcwNzU0MTQ.*_ga_CW55HF8NVT*MTY5OTE5NzQxOS4yMy4xLjE2OTkxOTc4MzIuNTEuMC4w",
                result:0
            }
        ]
    }
]

export default quiz