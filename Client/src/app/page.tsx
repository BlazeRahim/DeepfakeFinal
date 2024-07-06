import Image from "next/image";
import Home from '@/components/Home';
import Footer from '@/components/Footer';
import Deepfake from '@/components/Deepfake';
import Instruction from '@/components/instruction';
import Games from '@/components/Games';
import Creator from '@/components/creator';
import Features from '@/components/features'
import Beam from '@/components/downcontent/component'
export default function Aooo() {
  return (
    <div className="flex justify-center items-center overflow-hidden flex-col w-full">
      <Home />
      <Features />
      <Instruction />
      <Beam />
      <Deepfake />
      <Games />
      {/* <Creator/> */}
      <Footer />
    </div>
  );
}
