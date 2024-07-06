'use client';
import React, { useEffect, useState, useRef } from "react";
import { createFFmpeg, fetchFile } from "@ffmpeg/ffmpeg/src/index.js"
import * as helpers from "./utils/helpers";
import VideoFilePicker from "./videoFilePicker";
import OutputVideo from "./videoPlayer";
import OutputVideo2 from "./outputNoDown"
import RangeInput from "./videoRangeInput";
import { Button } from '@/components/ui/button'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip"


import { Highlighter, ArrowLeft, ScissorsLineDashed, CircleHelp } from 'lucide-react';
const FF = createFFmpeg({
  log: false,
  corePath: "https://unpkg.com/@ffmpeg/core@0.10.0/dist/ffmpeg-core.js",
});
function VideoApp({ fileImage, video, setExtractMeta, setPassedAudioDataUrl, setVideo, setVideoObject, setopen, setVideoUrl }) {
  const [inputVideoFile, setInputVideoFile] = useState(video);
  const [trimmedVideoFile, setTrimmedVideoFile] = useState(null);

  const [trimIsProcessing, setTrimIsProcessing] = useState(false);
  const [videoMeta, setVideoMeta] = useState(null);
  const [URL, setURL] = useState(null);
  const [rStart, setRstart] = useState(0); // 0%
  const [rEnd, setRend] = useState(20); // 10%
  const [thumbnails, setThumbnails] = useState([]);
  const [thumbnailIsProcessing, setThumbnailIsProcessing] = useState(false);

  const [show, setShow] = useState(true)
  const [showBtn, setShowBtn] = useState(true)

  const [saveBtn, setSaveBtn] = useState(false)


  const [videoAppShow, setVideoAppShow] = useState(true)
  const [addedShow, setAddedShow] = useState(true)
  const [deletedState, setDeletedState] = useState(true)
  const [backState, setBackState] = useState(false)

  const [mediaTitle, setMediaTitle] = useState("Untitled")

  const [isEditing, setIsEditing] = useState(false);
  const [clipTitle, setClipTitle] = useState('Your Clip Title');
  const [loadingText, setLoadingText] = useState("Loading...");
  const videoRef = useRef(null);

  const [processingType, setProcessingType] = useState('select')


  useEffect(() => {
    (async function () {
      if (!FF.isLoaded()) {
        await FF.load();
      }
    })();
  }, [])



  useEffect(() => {
    console.log(video)
    if (video !== null) {
      handleChange(video)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [video])

  const handleChange = async (file) => {
    // console.log(file);
    setDeletedState(true);
    const url = await readFileAsBase64(file);
    setURL(url);
  };

  async function readFileAsBase64(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();

      reader.onload = () => {
        resolve(reader.result);
      };

      reader.onerror = reject;

      reader.readAsDataURL(file);
    });
  }

  // useEffect(() => {
  //   if (videoRef.current) {
  //     videoRef.current.currentTime = rStart;
  //   }
  // }, [rStart]);

  const handleUpdateRange = (func) => {
    return ({ target: { value } }) => {
      func(value);
    };
  };

  const getThumbnails = async ({ duration }) => {
    if (!FF.isLoaded()) await FF.load();
    setThumbnailIsProcessing(true);
    let MAX_NUMBER_OF_IMAGES = 15;
    let offset =
      duration === MAX_NUMBER_OF_IMAGES ? 1 : duration / MAX_NUMBER_OF_IMAGES;
    let NUMBER_OF_IMAGES = duration < MAX_NUMBER_OF_IMAGES ? duration : 15;
    FF.FS("writeFile", inputVideoFile.name, await fetchFile(inputVideoFile));
    const arrayOfImageURIs = [];
    for (let i = 0; i < NUMBER_OF_IMAGES; i++) {
      let startTimeInSecs = helpers.toTimeString(Math.round(i * offset));
      if (startTimeInSecs + offset > duration && offset > 1) {
        offset = 0;
      }
      try {
        await FF.run(
          "-ss",
          startTimeInSecs,
          "-i",
          inputVideoFile.name,
          "-t",
          "00:00:1.000",
          "-vf",
          `scale=150:-1`,
          `img${i}.png`
        );
        const data = FF.FS("readFile", `img${i}.png`);
        let blob = new Blob([data.buffer], { type: "image/png" });
        let dataURI = await helpers.readFileAsBase64(blob);
        arrayOfImageURIs.push(dataURI);
        FF.FS("unlink", `img${i}.png`);
      } catch (error) {
        console.log({ message: error });
      }
    }
    // getFirstFrameImageURL();
    setThumbnailIsProcessing(false);
    return arrayOfImageURIs;
  };

  const getFirstFrameImageURL = async (file) => {
    if (!FF.isLoaded()) await FF.load();
    const startTimeInSecs = "00:00:00.005";
    FF.FS("writeFile", file.name, await fetchFile(file));

    try {
      console.log(`Generating the first frame`);
      await FF.run(
        "-ss",
        startTimeInSecs,
        "-i",
        file.name,
        "-t",
        "00:00:1.000",
        "-vf",
        "scale=1280:-1",
        "firstFrame.png"
      );
      const data = FF.FS("readFile", "firstFrame.png");
      let blob = new Blob([data.buffer], { type: "image/png" });
      // Convert the Blob to a data URL
      const dataURI = await new Promise((resolve) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve(reader.result);
        reader.readAsDataURL(blob);
      });
      FF.FS("unlink", "firstFrame.png");
      fileImage(dataURI)
    } catch (error) {
      console.error(`Error generating the first frame:`, error);
      return null;
    }
  };

  const handleLoadedData = async (e) => {
    const el = e.target;
    const meta = {
      name: inputVideoFile.name,
      duration: el.duration,
      videoWidth: el.videoWidth,
      videoHeight: el.videoHeight,
    };
    setVideoMeta(meta);
    const thumbnails = await getThumbnails(meta);
    setThumbnails(thumbnails);
  };

  // const toggleVideoApp = async () => {
  //   setVideoAppShow(false);
  //   setSaveBtn(true);
  //   setAddedShow(false);
  // }

  // const deleteAction = async () => {
  //   setVideoAppShow(true);
  //   setShow(true)
  //   setAddedShow(true);
  //   setInputVideoFile(null);
  //   setDeletedState(false);
  //   setTrimmedVideoFile(null);
  //   uploadProvided(true);
  //   setBackState(false)
  // }

  const handleAudioTrim = async () => {
    let startTime = ((rStart / 100) * videoMeta.duration).toFixed(2);
    let offset = ((rEnd / 100) * videoMeta.duration - startTime).toFixed(2);
    try {
      FF.FS("writeFile", inputVideoFile.name, await fetchFile(inputVideoFile));
      await FF.run(
        "-ss",
        helpers.toTimeString(startTime),
        "-i",
        inputVideoFile.name,
        "-map",
        "0:a:0",
        "-vn",
        "-t",
        helpers.toTimeString(offset),
        "-acodec",
        "libmp3lame",
        "audio.mp3"
      );
      const data = FF.FS("readFile", "audio.mp3");
      setPassedAudioDataUrl(data)
    } catch (error) {
      console.log(error);
    }
  };

  const setVideoSize = async (data) => {
    const videoSizeInBytes = new Blob([data.buffer], { type: "video/mp4" }).size;
    const videoNameFile = inputVideoFile.name
    const startTime = ((rStart / 100) * videoMeta.duration).toFixed(2);
    const offset = ((rEnd / 100) * videoMeta.duration - startTime).toFixed(2);
    const extractedMeta = {
      videoName: videoNameFile,
      videoDuration: (offset / 60).toFixed(0),
      videoDurationString: helpers.toTimeString(offset),
      videoSize: videoSizeInBytes,
    }
    setExtractMeta(extractedMeta)
  };

  const handleTrim = async () => {
    let startTime = ((rStart / 100) * videoMeta.duration).toFixed(2);
    let offset = ((rEnd / 100) * videoMeta.duration - startTime).toFixed(2);
    const offsetLenght = (offset / 60).toFixed(0);

    if (10 > offset > 5) {
      setLoadingText(`Your Video is ${(offset / 60).toFixed(0)} Minutes => Process Time May Take About 30 secounds`)
    } else if (offset > 10) {
      setLoadingText(`Your Video is ${(offset / 60).toFixed(0)} Minutes => Process Time May Take About 45 secounds`)
    } else if (offset < 5) {
      setLoadingText(`Your Video is ${(offset / 60).toFixed(0)} Minutes => Process Time May Take About 15 secounds`)
    } else {
      setLoadingText(`Loading...`)
    }
    setTrimIsProcessing(true);
    setShow(false);
    if (deletedState == false) {
      setDeletedState(!deletedState)
    }
    setShowBtn(false);
    setSaveBtn(true);
    try {
      FF.FS("writeFile", inputVideoFile.name, await fetchFile(inputVideoFile));
      await FF.run(
        "-ss",
        helpers.toTimeString(startTime),
        "-i",
        inputVideoFile.name,
        "-t",
        helpers.toTimeString(offset),
        "-c:v",
        "copy",
        "ping.mp4"
      );
      const data = FF.FS("readFile", "ping.mp4");
      console.log(data)
      await setVideoSize(data);
      await handleAudioTrim()
      const trimmedBlob = new Blob([data], { type: "video/mp4" });

      // Convert Blob to File object
      const trimmedVideoFile = new File([trimmedBlob], "trimmed_video.mp4", { type: "video/mp4" });

      // Set inputVideoFile state
      getFirstFrameImageURL(trimmedVideoFile)
      setInputVideoFile(trimmedVideoFile);

      // Set URL for the video
      const trimmedDataURL = window.URL.createObjectURL(trimmedBlob)
      setURL(trimmedDataURL);

      // const dataURL = await helpers.readFileAsBase64(
      //   new Blob([data.buffer], { type: "video/mp4" })
      // );
      // await handleAudioTrim()
      // console.log(dataURL)
      // setInputVideoFile(null)
      // setURL(null)
      // setInputVideoFile(blobToFile(dataURL, 'file_fromk.mp4'));
      // setURL(window.URL.createObjectURL(data));

    } catch (error) {
      console.log(error);
    } finally {
      setTrimIsProcessing(false);
    }
  };



  const removeAndMergeVideo = async () => {
    try {
      // Calculate the duration of the clip to be removed
      const clipStart = rStart / 100 * videoMeta.duration;
      const clipEnd = rEnd / 100 * videoMeta.duration;
      const durationBeforeClip = clipStart;
      const durationAfterClip = videoMeta.duration - clipEnd;
      const clipDuration = clipEnd - clipStart;

      if (10 > clipDuration > 5) {
        setLoadingText(`Your Video is ${(clipDuration / 60).toFixed(0)} Minutes => Process Time May Take About 30 secounds`)
      } else if (clipDuration > 10) {
        setLoadingText(`Your Video is ${(clipDuration / 60).toFixed(0)} Minutes => Process Time May Take About 45 secounds`)
      } else if (clipDuration < 5) {
        setLoadingText(`Your Video is ${(clipDuration / 60).toFixed(0)} Minutes => Process Time May Take About 15 secounds`)
      } else {
        setLoadingText(`Loading...`)
      }
      setTrimIsProcessing(true);
      setShow(false);
      if (deletedState == false) {
        setDeletedState(!deletedState)
      }
      setShowBtn(false);
      setSaveBtn(true);

      FF.FS("writeFile", inputVideoFile.name, await fetchFile(inputVideoFile));
      await FF.run(
        "-ss", "0",
        "-i", inputVideoFile.name,
        "-t", helpers.toTimeString(clipStart),
        "-c", "copy",
        "trimmed_start.mp4"
      );

      // Trim the video from the end to the end point
      await FF.run(
        "-ss", helpers.toTimeString(clipEnd),
        "-i", inputVideoFile.name,
        "-c", "copy",
        "trimmed_end.mp4"
      );

      const trimmedStartData = FF.FS("readFile", "trimmed_start.mp4");
      const trimmedEndData = FF.FS("readFile", "trimmed_end.mp4");

      if (!trimmedStartData || !trimmedEndData) {
        throw new Error("Trimmed video files not found.");
      }
      await FF.run(
        "-i", "trimmed_start.mp4",
        "-i", 'trimmed_end.mp4',
        "-filter_complex", "[0:v:0][1:v:0]concat=n=2:v=1:a=0[outv]",
        "-map", "[outv]",
        "merged_video.mp4"
      );

      const mergedData = FF.FS("readFile", "merged_video.mp4");


      if (!mergedData) {
        throw new Error("Merged video file not found.");
      }


      console.log(mergedData)
      await setVideoSize(mergedData);
      await handleAudioTrim()

      // Convert merged data to Blob
      const mergedBlob = new Blob([mergedData], { type: "video/mp4" });

      // Convert Blob to File object
      const mergedVideoFile = new File([mergedBlob], "merged_video.mp4", { type: "video/mp4" });
      getFirstFrameImageURL(mergedVideoFile)
      setInputVideoFile(mergedVideoFile);

      const mergedDataURL = window.URL.createObjectURL(mergedBlob)
      setURL(mergedDataURL);

    } catch (error) {
      console.error("Error removing and merging video:", error);
      throw error;
    } finally {
      setTrimIsProcessing(false);
    }
  };


  const handleEditClick = () => {
    setIsEditing(true);
  };

  const handleTitleChange = (event) => {
    setClipTitle(event.target.value);
  };

  const handleTitleSave = () => {
    setIsEditing(false);
    // You can save the edited title here, e.g., by making an API request.
  };

  const HandleBack = () => {
    setShow(!show)
    setDeletedState(!deletedState)
    setBackState(true)
    // setSaveBtn(false)
  }

  const handleMediaTitle = (event) => {
    setMediaTitle(event.target.value);
    // handleTitleInput(event.target.value)
  }

  function blobToFile(blob, fileName) {
    const file = new File([blob], fileName, { type: blob.type });
    return file;
  }

  const saveTrimVideoToMain = () => {
    // setVideoUrl(window.URL.createObjectURL(inputVideoFile))
    setVideoObject(inputVideoFile)
    const vdata = new FormData()
    vdata.append("file", inputVideoFile)
    console.log(vdata)
    setVideo(vdata)
    setShow(true)
    setSaveBtn(false)
    setShowBtn(true)
  }

  return (
    <div className="w-full flex flex-col gap-[50px]">
      {videoAppShow ? <main className="w-full flex flex-col gap-[20px]">


        <section className="flex flex-col items-center base:mt-[30px] bl:mt-[-10px] w-full">
          <article className="w-full  flex flex-col items-center justify-center">
            {show ?
              <div className="w-full flex justify-end items-center mb-[15px]">
                <div className="w-[100px] justify-center   flex items-center">
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <CircleHelp size={16} className="cursor-pointer ml-[40px]" color="blue" />
                      </TooltipTrigger>
                      <TooltipContent className="!max-w-[120px] w-[120px] text-center text-xs" side="left">
                        <p className="!max-w-[120px] !text-xs !text-center" >Choose whether to Remove clips or Select clips</p>
                      </TooltipContent>
                    </Tooltip>  
                  </TooltipProvider>
                </div>
                <Button onClick={(e) => {
                  if (processingType !== 'remove') setProcessingType('remove')
                }} className={`rounded-none px-[20px] rounded-l-lg ${processingType === 'remove' ? 'bg-primary' : 'bg-secondary hover:bg-secondary/30'} `}>Remove</Button>
                <Button onClick={(e) => {
                  if (processingType !== 'select') setProcessingType('select')
                }} className={`rounded-none px-[20px] rounded-r-lg ${processingType === 'select' ? 'bg-primary' : 'bg-secondary hover:bg-secondary/30'} `}>Select</Button>
              </div>
              : null
            }
            {trimIsProcessing ? <h4>{loadingText}</h4> : null}
            {show ?
              <>
              <div className="w-full base:hidden bl:flex h-[280px]  overflow-hidden">
                <video
                  ref={videoRef}
                  className='w-full h-full border object-fit'
                  src={inputVideoFile ? URL : null}
                  autoPlay
                  controls
                  muted
                  onLoadedMetadata={handleLoadedData}
                  crossOrigin="anonymous"
                  style={{ maxHeight: '280px' }}
                >
                </video>
              </div>

              <div className="w-full base:flex bl:hidden h-[230px]   overflow-hidden">
                <video
                  ref={videoRef}
                  className='w-full h-full border object-fit'
                  src={inputVideoFile ? URL : null}
                  autoPlay
                  controls
                  muted
                  onLoadedMetadata={handleLoadedData}
                  crossOrigin="anonymous"
                  style={{ maxHeight: '230px' }}
                >
                </video>
              </div>

              </>
              
              : null
            }

          </article>
        </section>

        {<div className="w-full relative">
          <RangeInput
            rEnd={rEnd}
            rStart={rStart}
            handleUpdaterStart={handleUpdateRange(setRstart)}
            handleUpdaterEnd={handleUpdateRange(setRend)}
            loading={thumbnailIsProcessing}
            videoMeta={videoMeta}
            thumbNails={thumbnails}
          />
        </div>}

      </main> : null
      }

      {showBtn ?
        <div className="flex items-center bl:min-w-[80px] bl:max-w-[200px] justify-center base:mt-[20px] bl:mt-[-5px] gap-[15px] w-full">
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant='outline'
                  className="px-[25px]"
                  onClick={(e) => {
                    e.preventDefault()
                    saveTrimVideoToMain()
                    setopen(false)
                  }}
                  disabled={trimIsProcessing || thumbnailIsProcessing}
                >
                  Save and continue
                </Button>
              </TooltipTrigger>
              <TooltipContent side="bottom">
                <p className="max-w-[200px] text-xs text-center">Complete the trimming and go to detect video</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  className="px-[25px]"
                  onClick={(e) => {
                    e.preventDefault()
                    if (processingType === 'select') handleTrim()
                    else if (processingType === 'remove') removeAndMergeVideo()
                  }}
                  disabled={trimIsProcessing}
                >
                  {thumbnailIsProcessing === false ? <ScissorsLineDashed className="mr-2 h-4 w-4" /> : " Loading..."}
                  {trimIsProcessing ? "Loading " :
                    "Clip"
                  }
                </Button>
              </TooltipTrigger>
              <TooltipContent side="bottom">
                <p className="max-w-[200px] text-xs text-center">Select the Clip from the above range <br />
                  and trim (remove) other parts</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div> : null
      }

      {saveBtn ?
        <div className="flex items-center justify-center gap-[25px] w-full">
          <Button variant='outline' onClick={HandleBack} className="flex justify-center items-center gap-[14px]">
            <ArrowLeft size={20} color="white" />
            Back
          </Button>
          <Button
            className="px-[25px]"
            onClick={(e) => {
              e.preventDefault()
              saveTrimVideoToMain()
            }}
            disabled={trimIsProcessing}
          >
            {trimIsProcessing ? "loading..." : "Save"}

          </Button>
        </div> : null
      }
    </div>
  );
}

export default VideoApp
