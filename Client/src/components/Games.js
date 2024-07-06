"use client";
import React, { useState, useEffect, useRef } from "react";
import "../scss/games.scss";
import { motion } from "framer-motion";
import scroll from "../utils/scroll";
import { X } from "lucide-react";
import Confetti from "react-confetti";
import ConfettiExplosion from "react-confetti-explosion";
import Quiz from "../utils/quiz";
import ArrowBackIosNewIcon from "@mui/icons-material/ArrowBackIosNew";
import ArrowForwardIosIcon from "@mui/icons-material/ArrowForwardIos";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb";
import {
  Carousel,
  CarouselContent,
  CarouselItem,
  CarouselNext,
  CarouselPrevious,
} from "@/components/ui/carousel";
import { ChevronRight } from "lucide-react";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { RotateCcw } from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
const Games = () => {
  const [data, setdata] = useState([]);
  const [flag, setflag] = useState(false);
  const [rule, setrule] = useState(false);
  const [correct, setcorrect] = useState(false);
  const box = useRef(null);
  const game = useRef(null);
  const [checked, setchecked] = useState(false);
  const [fast, setfast] = useState(false);
  const [finish, setfinish] = useState(false);
  const [score, setscore] = useState(0);
  const [edition, setedition] = useState(-1);
  const [final, setfinal] = useState(false);
  // const [isLoading, setisLoading] = useState(false);
  const [size, setSize] = useState(1000);
  const [idxx, setidxx] = useState(0);
  const [cards, setCardse] = useState([]);

  useEffect(() => {
    const screenWIdth = window.innerWidth;
    if (screenWIdth > 700) {
      setSize(1000);
    } else {
      setSize(screenWIdth - 100);
    }
  }, []);

  const valueData = [
    "Celebrity Edition",
    "Politician Edition",
    "Random YT Video",
  ];

  useEffect(() => {
    if (flag === false) {
      setflag(false);
      setrule(false);
      setchecked(false);
      setdata([]);
    }
  }, [flag]);

  useEffect(() => {
    if (correct === true) {
      setTimeout(() => {
        setcorrect(false);
      }, 3400);
    }
    // eslint-disable-next-line
  }, [correct]);

  useEffect(() => {
    if (rule === false && data.length !== 0) {
      let name = "#id0";
      const ele = document.querySelector(name);
      ele.style.display = "flex";
      for (let i = 1; i < data.length; i++) {
        let name = "#id" + i;
        const ele = document.querySelector(name);
        ele.style.display = "none";
      }
    }
    // eslint-disable-next-line
  }, [rule]);

  return (
    <div
      className="flex items-center justify-center flex-col relative bg-accent w-full"
      ref={game}
    >
      <div className="flex flex-col items-center py-[20px]  justify-start  min-w-[min(1300px,90%)] base:w-[100%] tab:w-[min(1300px , 90%)] base:max-w-[100%]  tab:max-w-[1300px] ">
        <motion.h2
          className="base:text-xl tab:text-3xl font-[440] py-[10px] w-full text-left base:px-[13px] bl:px-[40px]"
          initial={{ opacity: 0, x: -150 }}
          viewport={{ once: true }}
          whileInView={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.7, type: "Spring", bounce: 0.4 }}
        >
          DeepFake challenge
        </motion.h2>
        <div className="base:max-w-[100vw] tab:max-w-auto  tab:w-[80%]  bl:w-full h-[calc(100%_-_4rem)] py-[70px] flex justify-center items-center">
          <Carousel className="base:max-w-[75vw] tab:max-w-auto tab:max-w-auto ">
            <CarouselContent className="base:w-full tab:w-[100%]  bl:w-auto  max-w-[820px] min-h-[min(430px,90%)] tab:max-h-[500px]">
              {Quiz.map((item, idx) => (
                <CarouselItem
                  key={idx}
                  className="base:w-[110%] tab:w-[100%] bl:w-[770px]  min-h-[430px] !rounded-xl tab:max-h-[500px] flex  justify-center items-center"
                >
                  <div className="p-1 base:w-full tab:w-[90%]  bl:w-[770px] min-h-[430px] tab:max-h-[500px] !rounded-xl">
                    <Card className="base:w-full tab:w-[100%] bl:w-[770px] min-h-[430px] tab:max-h-[500px] !rounded-xl">
                      <CardContent className="flex base:w-full tab:w-[100%] bl:w-[770px] min-h-[430px] tab:max-h-[500px] !rounded-xl items-center gap-[30px] justify-center p-6 base:flex-col tab:flex-row base:py-[10px] tab:py-[0px]">
                        <img
                          crossOrigin="anonymous"
                          src={item.cover}
                          className="base:w-full tab:w-[45%] !bl:w-[320px] base:h-[140px] tab:h-[210px]"
                          alt="ef"
                        />
                        <div className="base:w-full tab:w-[55%] bl:w-[450px]  flex flex-col gap-6">
                          <p>
                            {item.heading}{" "}
                            <span className="text-primary">
                              {item.context}.
                            </span>
                          </p>
                          <p className="base:hidden tab:inline-block">
                            {item.focus}.
                          </p>
                          <Button
                            className="min-w-[80px] max-w-[130px] px-[15px]  py-[4px]"
                            onClick={(e) => {
                              e.preventDefault();
                              setflag(true);
                              setrule(true);
                              setdata(item.data);
                              setedition(item.edition);
                              setchecked(false);
                            }}
                          >
                            Start
                          </Button>
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                </CarouselItem>
              ))}
            </CarouselContent>
            <CarouselPrevious />
            <CarouselNext />
          </Carousel>
        </div>

        {flag === true && (
          <Dialog open={flag} onOpenChange={setflag}>
            <DialogContent className="base:min-w-[90%] base:max-w-[90%] bl:min-w-[min(1300px,80vw)] base:h-[90vh] tab:h-[80vh] bl:max-w-[min(1300px,80vw)] bl:h-[min(700px,78vh)] flex flex-col px-[0px] py-[0px] items-center !overflow-hidden">
              {correct && <ConfettiExplosion />}
              {correct && (
                <>
                  <Confetti className="confetti" />
                </>
              )}
              <div className="flex w-full py-[16px] px-[25px] gap-[15px] items-center">
                <img
                  src="./images/Vlogo.png"
                  alt="asd"
                  className="w-[55px] h-[55px]"
                />
                <Breadcrumb>
                  <BreadcrumbList>
                    <BreadcrumbItem>
                      <BreadcrumbLink
                        className="cursor-pointer"
                        onClick={(e) => setflag(false)}
                      >
                        Challenges
                      </BreadcrumbLink>
                    </BreadcrumbItem>
                    <BreadcrumbSeparator />
                    <BreadcrumbItem>
                      <BreadcrumbLink className="text-white cursor-pointer">
                        {valueData[edition]}
                      </BreadcrumbLink>
                    </BreadcrumbItem>
                  </BreadcrumbList>
                </Breadcrumb>
              </div>
              <img
                src="./images/tp2.png"
                alt=""
                className="absolute select-none z-0 bottom-0 base:right-[-20%] base:h-[50%] tab:right-[-9%]  tab:w-[290px] tab:h-[75%] base:opacity-[0.3] tab:opacity-[0.4]"
              />
              {rule === true ? (
                <div className="w-full h-[calc(100%_-_5rem)] flex justify-center items-center">
                  <div className="w-[75%] h-[90%] flex flex-col gap-[30px]">
                    <h1 className="text-xl font-[600]">How to Play -</h1>
                    <ul className="gap-2">
                      <li className="flex items-center gap-3">
                        <ChevronRight size={22} />
                        You will be shown a series of celebrities&quot; video
                        clips.
                      </li>
                      <li className="flex items-center gap-3">
                        <ChevronRight size={22} />
                        Your task is to determine if each video is real or a
                        deepfake.
                      </li>
                      <li className="flex items-center gap-3">
                        <ChevronRight size={22} />
                        Click on &quot;Real&quot; or &quot;Deepfake&quot; to
                        make your choice.
                      </li>
                      <li className="flex items-center gap-3">
                        <ChevronRight size={22} />
                        The authentic video for Every Deepfake edited video will
                        be shown after selecting an option.
                      </li>
                      <li className="flex items-center gap-3">
                        <ChevronRight size={22} />
                        No audio will be there , analyze the Face and make ur
                        guess.
                      </li>
                    </ul>
                    <Button
                      className="max-w-[100px]"
                      onClick={(e) => {
                        e.preventDefault();

                        setrule(false);
                      }}
                    >
                      less go !
                    </Button>
                  </div>
                </div>
              ) : (
                <>
                  {finish === false &&
                    data?.map((item, idx) => (
                      <motion.div
                        className="base:w-[90%] tab:w-[77%] base:h-[80%] tab:h-[70%] hidden base:mt-[20px] tab:mt-[-12px] gap-[50px] items-center justify-center base:flex-col tab:flex-row"
                        key={idx}
                        id={`id${idx}`}
                        initial={{ opacity: 0, x: size }}
                        viewport={{ once: true }}
                        whileInView={{ opacity: 1, x: 0 }}
                        transition={{
                          duration: 0.7,
                          type: "Spring",
                          bounce: 0.8,
                        }}
                      >
                        <div className="base:w-[100%] tab:w-[50%] base:h-[50%] tab:h-full flex flex-col items-center justify-center">
                          {fast === true && item.result === 0 && (
                            <div
                              className="w-full h-full flex flex-col gap-[20px] items-center justify-center"
                              initial={{ scale: 0 }}
                              viewport={{ once: true }}
                              whileInView={{ opacity: 1, scale: 1 }}
                              transition={{
                                duration: 4,
                                type: "Spring",
                                bounce: 0.9,
                              }}
                            >
                              <div className="ans">
                                <h1 className="text-[1.1rem] font-[440] mb-0">
                                  The video was DeepFake edit , Here is the
                                  Original video.
                                </h1>
                              </div>
                              <video
                                crossOrigin="anonymous"
                                className="w-full base:h-[85%] tab:h-[75%] border"
                                loop
                                autoPlay
                                controls
                              >
                                <source src={item.real} type="video/mp4" />
                                Your browser does not support the video tag.
                              </video>
                            </div>
                          )}
                          {fast === true && item.result === 1 && (
                            <motion.h1
                              className="text-[1.2rem] text-left font-[440] mb-2"
                              initial={{ scale: 0 }}
                              viewport={{ once: true }}
                              whileInView={{ opacity: 1, scale: 1 }}
                              transition={{ duration: 0.3 }}
                            >
                              The Video is Authentic
                            </motion.h1>
                          )}
                          {(fast === false || item.result === 1) && (
                            <video
                              className="w-full base:h-[85%] tab:h-[75%] border"
                              loop
                              autoPlay
                              controls
                            >
                              <source src={item.video} type="video/mp4" />
                              Your browser does not support the video tag.
                            </video>
                          )}
                        </div>
                        <div className="base:w-[100%] tab:w-[50%] base:h-[50%] tab:h-full  flex justify-center items-center">
                          {checked === false && (
                            <div
                              className="w-full h-full flex justify-center items-center flex-col gap-[25px]"
                              id={`upar${idx}`}
                            >
                              <Button
                                variant="secondary"
                                className="wrong optionbutton min-w-[140px] min-h-[60px]"
                                id={`wrong${idx}`}
                                onClick={(e) => {
                                  e.preventDefault();
                                  setfast(true);
                                  if (item.result === 0) {
                                    let tem = score;
                                    setscore(++tem);
                                    setcorrect(true);
                                    setchecked(true);
                                    setfinal(true);
                                  } else {
                                    let btn = document.querySelector(
                                      `#wrong${idx}`
                                    );
                                    btn.style.animation = "vibrate 0.7s ease";
                                    btn.addEventListener("animationend", () => {
                                      setchecked(true);
                                    });
                                  }
                                }}
                              >
                                Deepfake
                              </Button>
                              <Button
                                variant="secondary"
                                className="correct  optionbutton min-w-[140px] min-h-[60px]"
                                id={`correct${idx}`}
                                onClick={(e) => {
                                  e.preventDefault();
                                  setfast(true);
                                  if (item.result === 1) {
                                    let tem = score;
                                    setscore(++tem);
                                    setcorrect(true);
                                    setchecked(true);
                                    setfinal(true);
                                  } else {
                                    let btn = document.querySelector(
                                      `#correct${idx}`
                                    );
                                    btn.style.animation = "vibrate 0.7s ease";
                                    btn.addEventListener("animationend", () => {
                                      setchecked(true);
                                    });
                                  }
                                }}
                              >
                                Real
                              </Button>
                            </div>
                          )}
                          {checked === true && (
                            <div
                              className={`base:w-[90%] tab:w-[70%] base:h-[140px] tab:h-[180px] border flex flex-col gap-[20px] justify-center items-center ${
                                final === true
                                  ? "border-[#87cf7a]/40 bg-[#5cb85c]/10"
                                  : "border-destructive/40 bg-[#d9534f]/20"
                              } `}
                              initial={{ scale: 0 }}
                              viewport={{ once: true }}
                              whileInView={{ opacity: 1, scale: 1 }}
                              transition={{
                                duration: 0.6,
                                type: "Spring",
                                bounce: 0.9,
                              }}
                              id={final === true ? "correct" : "wrong"}
                            >
                              {final === true ? (
                                <h2 className="text-md font-[440]">
                                  You Guessed it Correct
                                </h2>
                              ) : (
                                <h2 className="text-md font-[440]">
                                  You Guess it Wrong
                                </h2>
                              )}
                              <Button
                                className={` min-w-[120px] min-h-[40px] ${
                                  final === true
                                    ? "bg-[#5cb85c] hover:bg-[#5cb85c]"
                                    : "bg-[#d9534f] hover:bg-[#c04b47]"
                                }`}
                                onClick={(e) => {
                                  e.preventDefault();
                                  let el1 = document.querySelector("#id" + idx);
                                  el1.style.position = "relative";
                                  el1.style.animation = "left_go 1s ease";
                                  el1.addEventListener("animationend", () => {
                                    el1.style.display = "none";
                                    let next = idx + 1;
                                    if (next < data.length) {
                                      let el2 = document.querySelector(
                                        "#id" + next
                                      );
                                      el2.style.display = "flex";
                                    } else {
                                      setfinish(true);
                                    }
                                    setchecked(false);
                                    setcorrect(false);
                                    setfast(false);
                                    setfinal(false);
                                  });
                                }}
                              >
                                Next Video
                              </Button>
                            </div>
                          )}
                        </div>
                      </motion.div>
                    ))}
                  {finish === true && (
                    <div className="base:w-[90%] tab:w-[70%] flex items-center flex-col gap-[43px]">
                      <motion.Card
                        className="base:w-[100%] tab:w-[500px] border"
                        x-chunk="dashboard-05-chunk-4"
                      >
                        <CardHeader className="flex flex-row items-start bg-muted/50">
                          <div className="grid gap-1.5">
                            <CardTitle className="group flex items-center gap-2 text-lg w-full text-[0.96rem] leading-[20px]">
                              Your Result
                            </CardTitle>
                            {score >= data.length ? (
                              <CardDescription className="text-xs">
                                You have guessed most of the videos correctly
                              </CardDescription>
                            ) : (
                              <CardDescription className="text-xs">
                                You have guessed very few videos correclty
                              </CardDescription>
                            )}
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
                                <span className="text-muted-foreground">
                                  Total Correct guess
                                </span>
                                <span>{score}</span>
                              </li>
                              <li className="flex items-center justify-between">
                                <span className="text-muted-foreground">
                                  Total wrong guess
                                </span>
                                <span>{data.length - score}</span>
                              </li>
                            </ul>
                          </div>
                        </CardContent>
                      </motion.Card>
                      <div className="base:w-[100%] tab:w-[500px] flex flex-col gap-[20px] ">
                        <h1 className="text-lg font-[550]">
                          Similar Challenges -
                        </h1>
                        <div className="flex items-center gap-[30px]">
                          {Quiz.filter((item) => item.edition != edition)
                            .slice(0, 2)
                            .map((qqqq, idex) => (
                              <TooltipProvider key={idex + 345}>
                                <Tooltip>
                                  <TooltipTrigger>
                                    <img
                                      src={qqqq.cover}
                                      alt="dsv"
                                      className="w-[120px] select-none h-[70px] rounded-lg border-[2px] border-[#5d5d5d]"
                                    />
                                  </TooltipTrigger>
                                  <TooltipContent side="bottom">
                                    <p>{qqqq.name}</p>
                                  </TooltipContent>
                                </Tooltip>
                              </TooltipProvider>
                            ))}
                        </div>
                      </div>
                    </div>
                  )}
                </>
              )}
            </DialogContent>
          </Dialog>
        )}
      </div>
    </div>
  );
};

export default Games;
