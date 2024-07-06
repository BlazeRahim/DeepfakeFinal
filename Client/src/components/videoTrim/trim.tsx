'use client'
import React, { useState } from 'react'
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogHeader,
    DialogTitle,
    DialogTrigger,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Scissors , X } from 'lucide-react'
import VideoTrim from './videoApp'
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
} from "@/components/ui/alert-dialog"
const Trim = ({ setThumbnail, video, setExtractMeta, setPassedAudioDataUrl, setVideo, setVideoObject, setVideoUrl }: {
    setThumbnail: any,
    video: any,
    setExtractMeta: any,
    setPassedAudioDataUrl: any,
    setVideo: any,
    setVideoObject: any,
    setVideoUrl: any,
}) => {
    const [open, setopen] = useState<boolean>(false)
    return (
        <AlertDialog open={open} onOpenChange={setopen} >
            <AlertDialogTrigger asChild>
                <Button onClick={(e) => {
                    setopen(true)
                }}
                    variant="outline"
                    className="flex justify-center px-[20px] items-center gap-[7px] text-[1.05rem] transition duration-500 ease hover:scale-105"
                >
                    <Scissors size={19} color="white" />
                    Trim
                </Button>
            </AlertDialogTrigger>
            <AlertDialogContent className="w-[100%]  min-h-[min(90vh,800px)] flex flex-col gap-[5px]">
                <div className="absolute right-4 top-4 rounded-sm opacity-70 ring-offset-background transition-opacity hover:opacity-100 focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:pointer-events-none data-[state=open]:bg-accent data-[state=open]:text-muted-foreground">
                    <X className="h-5 w-5 cursor-pointer" onClick={(e) => {
                        setopen(false)
                    }} />
                    <span className="sr-only">Close</span>
                </div>
                <AlertDialogHeader>
                    <AlertDialogTitle>Trim your video</AlertDialogTitle>
                    <AlertDialogDescription>
                        Remove unnecessary clips from the video and submit the trimmed version of video
                    </AlertDialogDescription>
                </AlertDialogHeader>
                <div className='flex w-full h-[calc(100%_-_4rem)] '>
                    <VideoTrim setVideoUrl={setVideoUrl} setopen={setopen} setVideo={setVideo} setVideoObject={setVideoObject} setPassedAudioDataUrl={setPassedAudioDataUrl} video={video} fileImage={setThumbnail} setExtractMeta={setExtractMeta} />
                </div>
            </AlertDialogContent>
        </AlertDialog>

    )
}

export default Trim