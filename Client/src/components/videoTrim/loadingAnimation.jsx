import * as React from 'react';
import { Loader2 } from 'lucide-react'

export default function LinearIndeterminate() {
  return (
    <div className='w-full'>
      <Loader2 className="mr-2 h-6 w-6 animate-spin" />
    </div>
  );
}