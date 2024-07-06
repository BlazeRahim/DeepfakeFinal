"use client";
import React, { useEffect, useState } from "react";
import "../scss/Home.scss";
import Scroll from "../utils/scroll";
import Navbar from "@/components/Navbar";
import { Button } from "./ui/button";
import { ArrowDown } from "lucide-react";
import { CircleChevronDown } from "lucide-react";
import Link from "next/link";
const Home = () => {
  return (
    <div className="home w-[100%] base:min-h-[100vh] bl:min-h-[min(100vh,750px)] bl:h-[min(100vh,750px)]  flex flex-col items-center ">
      <Navbar />
      <div className="min-w-[min(1300px,90%)]  w-[min(1300px , 90vw)]  relative h-[calc(100vh_-_5.5rem)] flex flex-col base:py-[65px] tab:py-[120px] gap-[40px]">
        <div className="flex flex-col gap-[27px] z-[100] base:px-[20px] tab:px-[0px] ">
          <h1 className="base:text-3xl tab:text-5xl font-[600]">
            Spot the Fake, Protect the Truth.
          </h1>
          <p className="base:text-2xl tab:text-4xl tab:w-[80%] font-[600] text-[#838383] tab:leading-[50px]">
            Utilizes eye <span className="text-primary">blinking</span>, facial{" "}
            <span className="text-primary">expressions</span>, <br /> and lip{" "}
            <span className="text-primary">syncing</span> to identify deepfakes.
          </p>
        </div>
        <img
          src="/images/landingtest.png"
          alt="meow"
          className="w-[1000px] select-none tab:flex base:hidden h-[600px] absolute top-[-20px] right-[-200px]"
        />
        <div className="flex items-center gap-[50px] base:px-[20px] base:py-[10px] tab:py-[5px] tab:px-[0px] ">
          <Button
            className="bg-white font-[600] base:hidden bl:flex text-primary text-[0.9rem] hover:bg-white rounded-[30px] min-w-[150px] 
          py-[18px] px-[10px] max-w-[160px] min-h-[50px] max-h-auto hover:bg-primary hover:text-white transition duration-200 ease "
            onClick={(e) => Scroll("deepfake")}
          >
            GET STARTED
          </Button>
          <Link href='/detect'
            className="bg-white font-[600] base:flex justify-center items-center bl:hidden text-primary text-[0.9rem]  rounded-[30px] min-w-[150px] 
          py-[18px] px-[10px] max-w-[160px] min-h-[50px] max-h-auto hover:bg-primary hover:text-white transition duration-200 ease "
          >
            GET STARTED
          </Link>
          <h2 onClick={(e) => Scroll("feature")} className="text-[#838383] flex hover:text-primary cursor-pointer items-center justify-center gap-2">
            Learn more <ArrowDown size={20} />
          </h2>
        </div>
        <div className="absolute bottom-[40px]  w-full flex justify-center items-center ">
          <CircleChevronDown
            size={28}
            color="white"
            className="hover:scale-105 cursor-pointer"
            onClick={(e) => {
              Scroll("feature");
            }}
          />
        </div>
      </div>
    </div>
  );
};

export default Home;
