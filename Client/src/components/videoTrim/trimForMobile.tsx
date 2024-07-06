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
import { Scissors, X } from 'lucide-react'
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
import {
    Tooltip,
    TooltipContent,
    TooltipProvider,
    TooltipTrigger,
} from "@/components/ui/tooltip"
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
        <>
            <TooltipProvider>
                <Tooltip>
                    <TooltipTrigger asChild>
                        <Button className='px-[16px] text-white hover:bg-card/40 flex justify-center items-center gap-2' variant='outline' onClick={(e) => {
                            e.preventDefault()
                            setopen(true)
                        }} >
                            <Scissors size={18} />
                        </Button>
                    </TooltipTrigger>
                    <TooltipContent>
                        <p>Trim Video</p>
                    </TooltipContent>
                </Tooltip>
            </TooltipProvider>
            <AlertDialog open={open} onOpenChange={setopen} >
                <AlertDialogContent className="w-[100%] overflow-y-auto overflow-x-hidden h-[88vh] flex flex-col gap-[10px]">

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
                    <div className='flex w-full h-[calc(100%_-_4rem)] mb-[20px]'>
                        <VideoTrim setVideoUrl={setVideoUrl} setopen={setopen} setVideo={setVideo} setVideoObject={setVideoObject} setPassedAudioDataUrl={setPassedAudioDataUrl} video={video} fileImage={setThumbnail} setExtractMeta={setExtractMeta} />
                    </div>
                </AlertDialogContent>
            </AlertDialog>
        </>

    )
}

export default Trim