'use client';
import React from "react";
import "../scss/creator.scss";
import { motion } from "framer-motion";
const Creator = () => {
  return (
    <div className="creator !w-[min(1400px , 85vw)]">
      <motion.h1
        initial={{ opacity: 0, x: -150 }}
        viewport={{ once: true }}
        whileInView={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.7, type: "Spring", bounce: 0.4 }}
      >
        Creators
      </motion.h1>
      <div className="box"></div>
      <div className="box"></div>
      <div className="box"></div>
      <div className="box"></div>
    </div>
  );
};
export default Creator;
