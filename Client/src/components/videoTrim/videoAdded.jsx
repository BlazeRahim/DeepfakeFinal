import * as React from 'react';
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Play } from 'lucide-react';
export default function MediaControlCard({ videoSrc }) {

  return (

    <Card>
      <CardContent>
        <Play aria-label='play/pause'/>
      </CardContent>
    </Card>
    // <Card sx={{ display: 'flex' }}>
    //   <Box sx={{ display: 'flex', flexDirection: 'column' }}>
    //     <CardContent sx={{ flex: '1 0 auto' }}>
    //       <Typography component="div" variant="h5">
    //         Live From Space
    //       </Typography>
    //       <Typography variant="subtitle1" color="text.secondary" component="div">
    //         Mac Miller
    //       </Typography>
    //     </CardContent>
    //     <Box sx={{ display: 'flex', alignItems: 'center', pl: 1, pb: 1 }}>

    //       <IconButton crossOrigin="anonymous" aria-label="play/pause">
    //         <PlayArrowIcon sx={{ height: 38, width: 38 }} />
    //       </IconButton>

    //     </Box>
    //   </Box>
    //   <CardMedia
    //     component="img"
    //     sx={{ width: 151 }}
    //     crossOrigin="anonymous"
    //     src={videoSrc}
    //     alt="Live from space album cover"
    //   />
    // </Card>
  );
}