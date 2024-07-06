"use client";
import './detect.scss'
import React, { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogHeader,
    DialogTitle,
    DialogTrigger,
    DialogFooter,
} from "@/components/ui/dialog";
import {
    Tooltip,
    TooltipContent,
    TooltipProvider,
    TooltipTrigger,
} from "@/components/ui/tooltip"
import {
    AlertDialog,
    AlertDialogAction,
    AlertDialogCancel,
    AlertDialogContent,
    AlertDialogDescription,
    AlertDialogFooter,
    AlertDialogHeader,
    AlertDialogTitle,
    AlertDialogTrigger,
} from "@/components/ui/alert-dialog";

import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { ColorRing } from "react-loader-spinner";
import { useToast } from "@/components/ui/use-toast";
import Image from "next/image";
import { RotateCcw, Copy, Loader2, X } from "lucide-react";
import { Separator } from "@/components/ui/separator";
import {
    Card,
    CardContent,
    CardDescription,
    CardFooter,
    CardHeader,
    CardTitle,
} from "@/components/ui/card";
import { Scissors, Trash2 } from "lucide-react";
import TrimComp from "@/components/videoTrim/trimForMobile";
import SocialComp from '@/components/chooseSocial/social'
import { getFirstFrameImageURL } from "@/utils/getFirstImage";

import Navbar from "@/components/Navbar";

const Deepfake = () => {
    const { toast } = useToast();
    const inputRef = useRef<any>(null);
    const [video, setvideo] = useState<any>(null);
    const [videoUrl, setVideoUrl] = useState<any>(null);
    const [flag, setflag] = useState<any>(false);
    const [reaction, setreaction] = useState<any>(-1);
    const color_code = ["#0ED23A", "#FF2A2A", "#FFB818"];
    const arr_emoji = [
        "./images/green.png",
        "./images/red.png",
        "./images/error.png",
    ];
    const [thumbnail, setThumbnail] = useState<any>(null);
    const [extractedMeta, setExtractMeta] = useState<any>(null);
    const [passedAudioData, setPassedAudioDataUrl] = useState<any>(null);
    const [temp, settemp] = useState<any>(videoUrl);
    const [api, setapi] = useState<any>(false);
    const abortcontroller = useRef<any>(null);
    const [result, setresult] = useState<any>(null);
    const [URL, setURL] = useState<any>("");
    const [getURLOpen, setGetURLOpen] = useState<any>(false);
    const [loadURLFetch, setLoadURLFetch] = useState<any>(false);
    const [linkName, setlinkName] = useState<any>("youtube");
    const [status, setStatus] = useState<any>(0);

    const [videoObject, setVideoObject] = useState<any>(null);

    useEffect(() => {
        console.log(video);
        if (video) {
            console.log("video yes");
            const element: any = document.querySelector(".down");
            if (element !== null) {
                element.style.display = "flex";
                element.style.borderTop = "3px dashed #bec0da;";
            }
        } else {
            console.log("video no");
            const element: any = document.querySelector(".down");
            if (element) {
                element.style.display = "none";
            }
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [video]);

    const handleClick = () => {
        // 👇️ open file input box on click of another element
        inputRef?.current.click();
    };

    const handleFileChange = async (event: any) => {
        const fileObj = event.target.files[0];
        if (!fileObj) {
            // setreaction(-1);
            return;
        }
        if (fileObj) {
            if (fileObj.size <= 30 * 1024 * 1024 && fileObj.type === "video/mp4") {
                console.log(fileObj);
                setVideoObject(fileObj);
                setVideoUrl(window.URL.createObjectURL(fileObj));

                const thumb = await getFirstFrameImageURL(fileObj);
                setThumbnail(thumb);

                const data = new FormData();
                data.append("file", fileObj);
                setvideo(data);
                const file = fileObj;
            } else {
                // setreaction(-1);
                alert("Please select an MP4 video file (max 30MB).");
            }
        }
        // console.log('fileObj is', fileObj);

        // 👇️ reset file input
        event.target.value = null;
    };

    useEffect(() => {
        if (flag === true && temp !== videoUrl) {
            settemp(videoUrl);
            const element2: any = document.querySelector(".imgMobile");
            if (element2 !== null) {
                element2.style.display = "flex";
                element2.style.animation = "increaseWidth 50s forwards";
            }
            const element3: any = document.querySelector(".imageMobile");
            if (element3 !== null) {
                element3.style.animation = "blink 2s infinite";
            }
        } else {
            const element2: any = document.querySelector(".img");
            if (element2 !== null) {
                element2.style.display = "none";
            }
            //  setreaction(-1);
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [flag]);

    const API = async (data: any) => {
        setapi(true);
        console.log(data);
        console.log("wennjdkfuihywbhdn");
        try {
            abortcontroller.current = new AbortController();
            const res = await fetch("http://localhost:5000/detect", {
                signal: abortcontroller.current.signal,
                method: "POST",
                headers: {
                    "X-Abort-Request": "true",
                },
                body: data,
            });
            const msg = await res.json();
            if (msg) {
                setStatus(msg.code);
                const element2: any = document.querySelector(".imgMobile");
                const element3: any = document.querySelector(".videowalaMobile");
                element2.style.animation = "restWidth 3s linear";
                element2.addEventListener(
                    "animationend",
                    function () {
                        element2.style.display = "none";
                        element3.style.animation = "none";
                        element3.style.animation = "autoScale 0.6s ease";
                        element3.style.borderRadius = "13px";
                        element3.style.border = `3px solid ${color_code[msg.code]}`;
                        setreaction(msg.code);
                    },
                    { once: true }
                );
                setTimeout(() => {
                    setresult(msg.result);
                    setapi(false)
                }, 3000)
            }
            console.log(msg);
        } catch (err) {
            console.log(err);
        }
    };

    useEffect(() => {
        if (reaction !== -1) {
            const element: any = document.querySelector(".videowalaMobile");
            // const rect = element.getBoundingClientRect();
            //  const relativeBottom = rect.top;
            //    const relativeRight = rect.left;
            const parentElement: any = document.querySelector("#ActivitySection");
            const elementRect: any = element.getBoundingClientRect();
            const parentRect: any = parentElement.getBoundingClientRect();

            const relativeTop = elementRect.top - parentRect.top;
            const relativeLeft = elementRect.left - parentRect.left;
            console.log(relativeTop);
            console.log(relativeLeft);
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [reaction]);

    const cancelrequest = () => {
        abortcontroller.current && abortcontroller.current.abort();
    };

    const URLFetch = async () => {
        setLoadURLFetch(true);
        try {
            if (URL.length === 0) {
                setLoadURLFetch(false);
                toast({
                    variant: "destructive",
                    title: "Empty Link",
                    description: "Please enter a valid URL",
                });
                return;
            }

            const response = await fetch("http://localhost:5000/getvideofromlink", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ video_url: URL, linkFrom: linkName }),
            });
            if (!response.ok) {
                toast({
                    variant: "destructive",
                    title: "Server Error",
                    description: "There is a issue in server",
                });
            }
            const videoBlob = await response.blob();
            const retrievedfile = blobToFile(videoBlob, "file_from_link.mp4");
            setVideoObject(retrievedfile);
            const videoUrlObject = window.URL.createObjectURL(videoBlob);
            setVideoUrl(videoUrlObject);

            const thumb = await getFirstFrameImageURL(retrievedfile);
            setThumbnail(thumb);

            const data = new FormData();
            data.append("file", videoBlob);
            setvideo(data);
            setLoadURLFetch(false);
            setGetURLOpen(false);
        } catch (err) {
            console.log(err);
            setLoadURLFetch(false);
            setGetURLOpen(false);
        }
    };

    useEffect(() => {
        console.log(videoUrl);
    }, [videoUrl]);

    function blobToFile(blob: any, fileName: string) {
        const file = new File([blob], fileName, { type: blob.type });
        return file;
    }

    useEffect(() => {
        if (result !== null) {
            console.log(result)
        }
    }, [result])

    return (
        <div className="base:flex bl:hidden w-[100%] flex-col relative items-center justify-center">
            <Navbar />
            <div className="w-full relative min-h-[calc(100vh_-_5.5rem)] h-[calc(100vh_-_5.5rem)] overflow-y-auto overflow-x-hidden flex flex-col">
                <Card className='w-full relative h-full'>
                    <CardHeader>
                        <CardTitle>Detect Deepfakes</CardTitle>
                        <CardDescription>Upload the video you want to verify and check it.</CardDescription>
                    </CardHeader>
                    <CardContent className='flex flex-col gap-[20px]'>
                        <div className='w-full py-[10px] border  rounded-xl  bg-muted flex justify-end items-center gap-4 px-[10px]'>
                            <input
                                hidden={true}
                                type="file"
                                id="Videoupload"
                                ref={inputRef}
                                accept="video/*"
                                onChange={handleFileChange}
                            />
                            {video === null && (
                                <>
                                    <AlertDialog open={getURLOpen} onOpenChange={setGetURLOpen}>
                                        <AlertDialogTrigger asChild>
                                            <Button className='px-[25px]'>Import</Button>
                                        </AlertDialogTrigger>
                                        <AlertDialogContent className="w-[min(92vw,400px)] rounded-xl  !bg-card">
                                            <div className="absolute right-4 top-4 rounded-sm opacity-70 ring-offset-background transition-opacity hover:opacity-100 focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:pointer-events-none data-[state=open]:bg-accent data-[state=open]:text-muted-foreground">
                                                <X className="h-5 w-5 cursor-pointer" onClick={(e) => {
                                                    setGetURLOpen(false)
                                                }} />
                                                <span className="sr-only">Close</span>
                                            </div>
                                            <AlertDialogHeader>
                                                <AlertDialogTitle>Enter Video URL</AlertDialogTitle>
                                                <AlertDialogDescription>
                                                    Make sure to upload a link which is public to everyone. and
                                                    size ot more than 30 mb.
                                                </AlertDialogDescription>
                                            </AlertDialogHeader>
                                            <div className="grid gap-4 py-4">
                                                <div className="grid grid-cols-4 items-center gap-4">
                                                    <Label htmlFor="name" className="text-right">
                                                        URL
                                                    </Label>
                                                    <Input
                                                        value={URL}
                                                        onChange={(e) => setURL(e.target.value)}
                                                        id="name"
                                                        className="col-span-3"
                                                    />
                                                </div>
                                            </div>
                                            <AlertDialogFooter className="gap-[25px] flex flex-row justify-end ">
                                                <SocialComp setlinkName={setlinkName} />

                                                <Button
                                                    disabled={loadURLFetch}
                                                    className="flex justify-center mt-5 items-center gap-1"
                                                    onClick={(e) => {
                                                        e.preventDefault();
                                                        URLFetch();
                                                    }}
                                                >
                                                    {loadURLFetch === true && (
                                                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                                    )}
                                                    Upload
                                                </Button>
                                            </AlertDialogFooter>
                                        </AlertDialogContent>
                                    </AlertDialog>
                                    <Button className='px-[25px]' onClick={(e) => {
                                        e.preventDefault();
                                        setvideo(null);
                                        setVideoUrl(null);
                                        setVideoObject(null)
                                        setflag(false);
                                        setreaction(-1);
                                        handleClick();
                                        setresult(null);
                                        setThumbnail(null);
                                        e.stopPropagation();
                                    }}>
                                        Upload
                                    </Button>
                                </>
                            )}
                            {video !== null && api === false && (
                                <>
                                    <Button className='px-[25px]' onClick={(e) => {
                                        API(video)
                                        setflag(true)
                                    }}>
                                        Detect
                                    </Button>
                                    <TrimComp
                                        setVideo={setvideo}
                                        setVideoObject={setVideoObject}
                                        setPassedAudioDataUrl={setPassedAudioDataUrl}
                                        video={videoObject}
                                        setThumbnail={setThumbnail}
                                        setExtractMeta={setExtractMeta}
                                        setVideoUrl={setVideoUrl}
                                    />
                                    <TooltipProvider>
                                    <Tooltip>
                                        <TooltipTrigger asChild>
                                            <Button className='px-[16px] text-destructive hover:bg-card/40 flex justify-center items-center gap-2' variant='outline' onClick={(e) => {
                                                e.preventDefault()
                                                setvideo(null);
                                                setVideoUrl(null);
                                                setflag(false);
                                                setVideoObject(null)
                                                setreaction(-1);
                                                setresult(null);
                                                setThumbnail(null);
                                            }} >
                                                <Trash2 size={18} />
                                            </Button>
                                        </TooltipTrigger>
                                        <TooltipContent>
                                            <p>Remove</p>
                                        </TooltipContent>
                                    </Tooltip>
                                </TooltipProvider>
                        </>
                            )}
                        {api === true && (
                            <Button variant="outline"
                                className="border bg-transparent border-primary text-primary  transition duration-500 ease hover:bg-primary hover:text-white px-[25px]"
                                onClick={(e) => {
                                    e.preventDefault();
                                    cancelrequest();
                                    setvideo(null);
                                    setVideoUrl(null);
                                    setflag(false);
                                    setVideoObject(null)
                                    setThumbnail(null)
                                    setreaction(-1);
                                    setresult(null)
                                    setapi(false);
                                }}>
                                Cancel
                            </Button>
                        )}
                    </div>
                    <div className='w-full flex flex-col py-[10px] gap-[20px]' id='ActivitySection'>
                        {thumbnail && (
                            <>
                                <motion.img
                                    initial={{ scale: 0 }}
                                    viewport={{ once: true }}
                                    whileInView={{ opacity: 1, scale: 1 }}
                                    transition={{ duration: 1 }}
                                    src={thumbnail}
                                    alt="thumbnail"
                                    className="videowalaMobile relative border opacity-1 w-full rounded-lg !h-[210px] object-contain bg-background z-1"
                                />
                                {reaction !== -1 && (
                                    <div className="w-full z-[10000] h-[50px] flex justify-end mt-[-70px] mr-[-25px]">
                                        <img src={arr_emoji[status]} alt="fv" className="w-[85px] h-[85px]" />
                                    </div>
                                )}
                                <div className="imageMobile">
                                    <img src="./images/load.png" alt="" className="imgMobile hidden" id="immg" />
                                </div>
                            </>
                        )}
                        {result && (
                            <Card
                                className="w-full rounded-lg"
                            >
                                <CardHeader className="flex flex-row items-start bg-muted/50">
                                    <div className="grid gap-1.5">
                                        <CardTitle
                                            className="group flex items-center gap-2 text-lg w-full text-[0.96rem] leading-[26px]"
                                            style={
                                                result.Faces === 0
                                                    ? { color: "#ff3333", opacity: 1 }
                                                    : { fontSize: "1.3rem" }
                                            }
                                        >
                                            {result.message}
                                            <Button
                                                size="icon"
                                                variant="outline"
                                                className="h-6 w-6 opacity-0 transition-opacity group-hover:opacity-100"
                                            >
                                                <Copy className="h-3 w-3" />
                                                <span className="sr-only">copy</span>
                                            </Button>
                                        </CardTitle>
                                        <CardDescription className="text-xs">
                                            As Veriface is in beta, the results aren't absolute truth ,
                                            don't consider this as an evidence.
                                        </CardDescription>
                                    </div>
                                    <div className="ml-auto flex items-center gap-1">
                                        <Button
                                            size="sm"
                                            variant="outline"
                                            onClick={(e) => {
                                                e.preventDefault();
                                                setresult(null);
                                                API(video);
                                                setflag(true);
                                                setreaction(-1);
                                            }}
                                            className="h-8 gap-1"
                                        >
                                            <RotateCcw className="h-3.5 w-3.5" />
                                            <span className="lg:sr-only xl:not-sr-only xl:whitespace-nowrap">
                                                Retry
                                            </span>
                                        </Button>
                                    </div>
                                </CardHeader>
                                <CardContent className="p-6 text-sm bg-card">
                                    <div className="grid gap-3">
                                        <div className="font-semibold">Details</div>
                                        <ul className="grid gap-3">
                                            <li className="flex items-center justify-between">
                                                <span className="text-muted-foreground">Total Frames</span>
                                                <span>{result.Frames}</span>
                                            </li>
                                            <li className="flex items-center justify-between">
                                                <span className="text-muted-foreground">
                                                    Total Faces in those frames
                                                </span>
                                                <span>{result.Faces}</span>
                                            </li>
                                        </ul>
                                        <Separator className="my-2" />
                                        <ul className="grid gap-3">
                                            <li className="flex items-center justify-between">
                                                <span className="text-muted-foreground">
                                                    Total Deepfake Faces %
                                                </span>
                                                <span>{result.Deepfake.toFixed(2)}%</span>
                                            </li>
                                            <li className="flex items-center justify-between">
                                                <span className="text-muted-foreground">
                                                    Total Real Faces %
                                                </span>
                                                <span>{result.Real.toFixed(2)}%</span>
                                            </li>
                                        </ul>
                                    </div>
                                </CardContent>
                            </Card>
                        )}
                    </div>
                </CardContent>
            </Card>

        </div>
            {/* <div
                className="left h-full !gap-[45px]  flex flex-col justify-center items-center w-[45%] "
                id="left"
            >
                {thumbnail ? (
                    <motion.img
                        initial={{ scale: 0 }}
                        viewport={{ once: true }}
                        whileInView={{ opacity: 1, scale: 1 }}
                        transition={{ duration: 1 }}
                        src={thumbnail}
                        alt="thumbnail"
                        className="videowala relative opacity-1 !w-[min(500px,90%)] rounded-lg !h-[300px] object-cover z-1 mt-[0]"
                    />
                ) : (
                    <div
                        id="helloo"
                        className="w-[min(500px,90%)] !h-[300px] z-[1000] 
          border-[3px] border-dashed border-primary/10 rounded-lg  flex justify-center items-center flex-col text-[0.9rem] !bg-muted/80"
                        onClick={(e) => {
                            e.preventDefault();
                            setvideo(null);
                            setVideoUrl(null);
                            setflag(false);
                            setreaction(-1);
                            handleClick();
                            setresult(null);
                            setThumbnail(null);
                            e.stopPropagation();
                        }}
                    >
                        <p className="child mb-[0.3rem] mt-[0.3rem] transition duration-100 ease-in text-[0.9rem] opacity-[0.8] ">
                            Upload your Videos
                        </p>
                        <p className="child mb-[0.3rem] mt-[0.3rem] transition duration-100 ease-in text-[0.9rem] opacity-[0.8]">
                            Upto 30 mb of video & mp4 format only!
                        </p>
                    </div>
                )}
                {reaction !== -1 && (
                    <div className="w-[550px] z-[10000]  h-[100px] flex justify-end mt-[-100px] mr-[-75px]">
                        <img src={arr_emoji[status]} alt="fv" className="react" />
                    </div>
                )}
                <div className="image">
                    <img src="./images/load.png" alt="" className="img" id="immg" />
                </div>
                {reaction === -1 && (
                    <div
                        className="w-[min(500px,90%)] !h-[300px] z-[100] 
          border-[3px] border-dashed border-primary/10 rounded-lg flex justify-center items-center flex-col text-[0.9rem] !bg-muted/80"
                    >
                        <p>Result will be displayed here.</p>
                    </div>
                )}
                {result && (
                    <Card
                        className="overflow-hidden w-[500px] border rounded-lg mt-[-48px]"
                        x-chunk="dashboard-05-chunk-4"
                    >
                        <CardHeader className="flex flex-row items-start bg-muted/50">
                            <div className="grid gap-1.5">
                                <CardTitle
                                    className="group flex items-center gap-2 text-lg w-full text-[0.96rem] leading-[20px]"
                                    style={
                                        result.Faces === 0
                                            ? { color: "#ff3333", opacity: 1 }
                                            : { fontSize: "1.3rem" }
                                    }
                                >
                                    {result.message}
                                    <Button
                                        size="icon"
                                        variant="outline"
                                        className="h-6 w-6 opacity-0 transition-opacity group-hover:opacity-100"
                                    >
                                        <Copy className="h-3 w-3" />
                                        <span className="sr-only">copy</span>
                                    </Button>
                                </CardTitle>
                                <CardDescription className="text-xs">
                                    As Veriface is in beta, the results aren't absolute truth ,
                                    don't consider this as an evidence.
                                </CardDescription>
                            </div>
                            <div className="ml-auto flex items-center gap-1">
                                <Button
                                    size="sm"
                                    variant="outline"
                                    onClick={(e) => {
                                        e.preventDefault();
                                        setresult(null);
                                        API(video);
                                        setflag(true);
                                        setreaction(-1);
                                    }}
                                    className="h-8 gap-1"
                                >
                                    <RotateCcw className="h-3.5 w-3.5" />
                                    <span className="lg:sr-only xl:not-sr-only xl:whitespace-nowrap">
                                        Retry
                                    </span>
                                </Button>
                            </div>
                        </CardHeader>
                        <CardContent className="p-6 text-sm bg-card">
                            <div className="grid gap-3">
                                <div className="font-semibold">Details</div>
                                <ul className="grid gap-3">
                                    <li className="flex items-center justify-between">
                                        <span className="text-muted-foreground">Total Frames</span>
                                        <span>{result.Frames}</span>
                                    </li>
                                    <li className="flex items-center justify-between">
                                        <span className="text-muted-foreground">
                                            Total Faces in those frames
                                        </span>
                                        <span>{result.Faces}</span>
                                    </li>
                                </ul>
                                <Separator className="my-2" />
                                <ul className="grid gap-3">
                                    <li className="flex items-center justify-between">
                                        <span className="text-muted-foreground">
                                            Total Deepfake Faces %
                                        </span>
                                        <span>{result.Deepfake.toFixed(2)}%</span>
                                    </li>
                                    <li className="flex items-center justify-between">
                                        <span className="text-muted-foreground">
                                            Total Real Faces %
                                        </span>
                                        <span>{result.Real.toFixed(2)}%</span>
                                    </li>
                                </ul>
                            </div>
                        </CardContent>
                    </Card>
                )}
            </div> */}

    {/* <div className="right min-w-[45%]  w-[45%] h-full relative flex flex-col justify-center items-center gap-[100px] ">
                <div className="w-full flex justify-center items-center">
                    <div className="h-[140px] w-[55%] px-[15px] py-[10px] bg-muted/80 border-dashed border-[3px] border-primary/10 rounded-lg flex justify-center items-center opacity-[0.5] ">
                        <p className="text-xs text-center">
                            Veriface aims to give an opinion about the scanned video and is
                            not responsible for the result. As Veriface is still in beta, the
                            results should not be treated as an absolute truth or evidence.
                        </p>
                    </div>
                </div>

                <div className="box !w-[min(400px,80%)] h-[360px] flex justify-evenly flex-col items-center mt-[-30px] bg-card !border-[2px] rounded-3xl">
                    <motion.div
                        className="up !gap-5"
                        initial={{ scale: 0 }}
                        viewport={{ once: true }}
                        whileInView={{ opacity: 1, scale: 1 }}
                    >
                        <Button
                            className="px-[35px] py-[25px] rounded-[30px] text-[1.15rem] transition duration-300 ease hover:scale-105"
                            id="uploaduu"
                            onClick={(e) => {
                                e.preventDefault();
                                setvideo(null);
                                setVideoUrl(null);
                                setflag(false);
                                setreaction(-1);
                                handleClick();
                                setresult(null);
                            }}
                        >
                            Upload your Videos
                        </Button>
                        <input
                            type="file"
                            id="Videoupload"
                            ref={inputRef}
                            accept="video/*"
                            onChange={handleFileChange}
                        />
                        <p>Upto 30 mb of video & mp4 format only!</p>
                        <AlertDialog open={getURLOpen} onOpenChange={setGetURLOpen}>
                            <AlertDialogTrigger asChild>
                                <p
                                    onClick={(e) => {
                                        e.preventDefault();
                                        setGetURLOpen(true);
                                    }}
                                    className="text-[0.8rem] underline cursor-pointer"
                                >
                                    Enter URL
                                </p>
                            </AlertDialogTrigger>
                            <AlertDialogContent className="sm:max-w-[425px] !bg-card">
                                <div className="absolute right-4 top-4 rounded-sm opacity-70 ring-offset-background transition-opacity hover:opacity-100 focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:pointer-events-none data-[state=open]:bg-accent data-[state=open]:text-muted-foreground">
                                    <X className="h-5 w-5 cursor-pointer" onClick={(e) => {
                                        setGetURLOpen(false)
                                    }} />
                                    <span className="sr-only">Close</span>
                                </div>
                                <AlertDialogHeader>
                                    <AlertDialogTitle>Enter Video URL</AlertDialogTitle>
                                    <AlertDialogDescription>
                                        Make sure to upload a link which is public to everyone. and
                                        size ot more than 30 mb.
                                    </AlertDialogDescription>
                                </AlertDialogHeader>
                                <div className="grid gap-4 py-4">
                                    <div className="grid grid-cols-4 items-center gap-4">
                                        <Label htmlFor="name" className="text-right">
                                            URL
                                        </Label>
                                        <Input
                                            value={URL}
                                            onChange={(e) => setURL(e.target.value)}
                                            id="name"
                                            className="col-span-3"
                                        />
                                    </div>
                                </div>
                                <AlertDialogFooter className="gap-[25px]">
                                    <SocialComp setlinkName={setlinkName} />
                                
                                    <Button
                                        disabled={loadURLFetch}
                                        className="flex justify-center mt-5 items-center gap-1"
                                        onClick={(e) => {
                                            e.preventDefault();
                                            URLFetch();
                                        }}
                                    >
                                        {loadURLFetch === true && (
                                            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                        )}
                                        Upload
                                    </Button>
                                </AlertDialogFooter>
                            </AlertDialogContent>
                        </AlertDialog>
                    </motion.div>
                    <motion.div
                        className="down border-t-[2px] border-dashed"
                        initial={{ scale: 0 }}
                        viewport={{ once: true }}
                        whileInView={{ opacity: 1, scale: 1 }}
                    >
                        {api === false && (
                            <div className="flex flex-col w-full gap-[20px] items-center justify-center">
                                <div className="flex gap-[20px] justify-between items-center">
                                    <TrimComp
                                        setVideo={setvideo}
                                        setVideoObject={setVideoObject}
                                        setPassedAudioDataUrl={setPassedAudioDataUrl}
                                        video={videoObject}
                                        setThumbnail={setThumbnail}
                                        setExtractMeta={setExtractMeta}
                                        setVideoUrl={setVideoUrl}
                                    />
                                    <Button
                                        variant="outline"
                                        className="flex text-destructive px-[20px]  hover:text-destructive  justify-center items-center gap-[7px] text-[1.05rem] transition duration-500 ease hover:scale-105"
                                        onClick={(e) => {
                                            e.preventDefault();
                                        }}
                                    >
                                        <Trash2 size={19} />
                                        Remove
                                    </Button>
                                </div>
                                <Button
                                    variant="outline"
                                    className="w-[70%] hover:bg-primary hover:text-white text-[0.96rem] transition duration-500 ease hover:scale-105"
                                    onClick={(e) => {
                                        e.preventDefault();
                                        API(video);
                                        const ele:any = document.querySelector(".up");
                                        const own:any = document.querySelector(".down");
                                        own.style.borderTop = "none";
                                        ele.style.display = "none";
                                        setflag(true);
                                    }}
                                >
                                    Detect Video
                                </Button>
                            </div>
                        )}
                        {api === true && (
                            <>
                                <p>This may take a few Seconds....</p>
                                <p>Estimated Time: 30-40 sec</p>
                                <Button
                                    variant="outline"
                                    className="mt-[2rem] w-[40%] rounded-[30px] border-[1.5px] bg-transparent border-primary text-primary text-[1.15rem] transition duration-500 ease hover:bg-primary hover:text-white"
                                    onClick={(e) => {
                                        e.preventDefault();
                                        cancelrequest();
                                        setvideo(null);
                                        setVideoUrl(null);
                                        setflag(false);
                                        setreaction(-1);
                                        setapi(false);
                                        const ele:any = document.querySelector(".up");
                                        ele.style.display = "flex";
                                    }}
                                >
                                    Cancel
                                </Button>
                            </>
                        )}
                    </motion.div>
                </div>
            </div> */}
    {/* <img src="./images/Polygon 1.png" alt="v" className="ploy" />
            <img src="./images/rope.png" alt="rve" className="rope" /> */}
        </div >
    );
};
export default Deepfake;
