"use client";

import { FaTwitter } from "react-icons/fa";
import { cn } from "@/lib/utils";
import { AnimatedBeam } from "@/components/magicui/animated-beam";
import React, { forwardRef, useRef } from "react";
import { Forward } from "lucide-react";
import Image from 'next/image'
export default function AnimatedBeamMultipleInputDemo() {
    const containerRef = useRef<HTMLDivElement>(null);
    const div1Ref = useRef<HTMLDivElement>(null);
    const div2Ref = useRef<HTMLDivElement>(null);
    const div3Ref = useRef<HTMLDivElement>(null);
    const div4Ref = useRef<HTMLDivElement>(null);
    const div5Ref = useRef<HTMLDivElement>(null);
    const div6Ref = useRef<HTMLImageElement>(null);
    const div7Ref = useRef<HTMLDivElement>(null);

    return (
        <div
            className="relative flex tab:h-full base:w-full tab:w-[50%]  items-center justify-center overflow-hidden rounded-lg  bg-[#2a2a2a] p-10 "
            ref={containerRef}
        >
            <div className="flex h-full w-full flex-row items-stretch justify-between gap-10">
                <div className="flex flex-col justify-center gap-3">
                    <div ref={div1Ref} className="h-12 w-12 z-10 flex  items-center justify-center rounded-full border-2 border-border bg-white p-3 shadow-[0_0_20px_-12px_rgba(0,0,0,0.8)]">
                        <Image src='/images/youtubeBeam.png' alt='sdcfv' width={32} height={32} />
                    </div>
                    <div ref={div2Ref} className="h-12 w-12 z-10 flex  items-center justify-center rounded-full border-2 border-border bg-white p-3 shadow-[0_0_20px_-12px_rgba(0,0,0,0.8)]">
                        <Image src='/images/instagramBeam.png' alt='sdcfv' width={32} height={32} />
                    </div>
                    <div ref={div3Ref} className="h-12 w-12 z-10 flex  items-center justify-center rounded-full border-2 border-border bg-white p-3 shadow-[0_0_20px_-12px_rgba(0,0,0,0.8)]">
                        <Image src='/images/twitterBeam.png' alt='sdcfv' width={32} height={32} />
                    </div>
                    <div ref={div4Ref} className="h-12 w-12 z-10 flex  items-center justify-center rounded-full border-2 border-border bg-white p-3 shadow-[0_0_20px_-12px_rgba(0,0,0,0.8)]">
                        <Image src='/images/driveBeam.png' alt='sdcfv' width={32} height={32} />
                    </div>
                    <div ref={div5Ref} className="h-12 w-12 z-10 flex  items-center justify-center rounded-full border-2 border-border bg-white p-3 shadow-[0_0_20px_-12px_rgba(0,0,0,0.8)]">
                        <Image src='/images/facebookBeam.png' alt='sdcfv' width={32} height={32} />
                    </div>
                </div>
                <div className="flex flex-col justify-center">
                    <Image ref={div6Ref} src='/images/logoBeam.png' className="z-[100000]" alt='sdcfv' width={54} height={54} />
                    {/* <div ref={div6Ref} className="h-14 w-14 z-10 flex  items-center justify-center rounded-full border-2 border-border bg-white p-3 shadow-[0_0_20px_-12px_rgba(0,0,0,0.8)]">
                        <Image src='/images/Vlogo.png' alt='sdcfv' width={45} height={45} />
                    </div> */}
                </div>
                <div className="flex flex-col justify-center">
                    <div ref={div7Ref} className="h-12 w-12 z-10 flex  items-center justify-center rounded-full border-2 border-border bg-white p-3 shadow-[0_0_20px_-12px_rgba(0,0,0,0.8)]">
                        <Image src='/images/botBeam.png' alt='sdcfv' width={36} height={36} />
                    </div>
                </div>
            </div>

            <AnimatedBeam
                containerRef={containerRef}
                fromRef={div1Ref}
                toRef={div6Ref}
            />
            <AnimatedBeam
                containerRef={containerRef}
                fromRef={div2Ref}
                toRef={div6Ref}
            />
            <AnimatedBeam
                containerRef={containerRef}
                fromRef={div3Ref}
                toRef={div6Ref}
            />
            <AnimatedBeam
                containerRef={containerRef}
                fromRef={div4Ref}
                toRef={div6Ref}
            />
            <AnimatedBeam
                containerRef={containerRef}
                fromRef={div5Ref}
                toRef={div6Ref}
            />
            <AnimatedBeam
                containerRef={containerRef}
                fromRef={div6Ref}
                toRef={div7Ref}
            />
        </div>
    );
}
