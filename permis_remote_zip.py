"""Seek the official TAO ZIP using verified local metadata and bounded HTTP ranges."""
from collections import OrderedDict
import hashlib
import io
import json
from pathlib import Path
import time
import urllib.request


class RemoteZipReader(io.RawIOBase):
    def __init__(self,status_file):
        super().__init__()
        status=json.loads(Path(status_file).read_text())
        assert status["status"]=="completed"
        self.url=status["source_url"]
        self.length=status["archive_total_bytes"]
        plan=json.loads((Path(status["central_directory_file"]).parent/"plan.json").read_text())
        self.regions=[(plan["central_directory_offset"],Path(status["central_directory_file"])),
                      (status["tail_offset"],Path(status["tail_file"]))]
        for _,path in self.regions:
            key="central_directory_sha256" if path.name=="central-directory.bin" else "tail_sha256"
            assert hashlib.sha256(path.read_bytes()).hexdigest()==status[key]
        self.position=0
        self.block_size=512*1024
        self.cache=OrderedDict()
        self.network_bytes=0
        self.network_requests=0

    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        pos=offset if whence==0 else self.position+offset if whence==1 else self.length+offset
        if whence not in (0,1,2) or pos<0:raise ValueError("Invalid ZIP seek")
        self.position=pos
        return pos

    def _fetch_block(self,start):
        end=min(start+self.block_size,self.length)-1
        for attempt in range(3):
            try:
                request=urllib.request.Request(self.url,headers={"Range":f"bytes={start}-{end}","Accept-Encoding":"identity"})
                with urllib.request.urlopen(request,timeout=45) as response:
                    assert response.status==206 and response.headers["Content-Range"]==f"bytes {start}-{end}/{self.length}"
                    data=response.read()
                    assert len(data)==end-start+1
                self.network_bytes+=len(data);self.network_requests+=1
                return data
            except Exception:
                if attempt==2:raise
                time.sleep(2)

    def read(self,size=-1):
        if self.position>=self.length:return b""
        if size is None or size<0:size=self.length-self.position
        size=min(size,self.length-self.position)
        if size>256*1024*1024:raise ValueError("Refusing an unbounded archive read")
        if not size:return b""
        for offset,path in self.regions:
            if offset<=self.position and self.position+size<=offset+path.stat().st_size:
                with path.open("rb") as stream:
                    stream.seek(self.position-offset);data=stream.read(size)
                assert len(data)==size
                self.position+=size
                return data
        parts=[];remaining=size
        while remaining:
            start=self.position//self.block_size*self.block_size
            if start not in self.cache:
                self.cache[start]=self._fetch_block(start)
                if len(self.cache)>32:self.cache.popitem(last=False)
            self.cache.move_to_end(start)
            data=self.cache[start]
            within=self.position-start
            count=min(remaining,len(data)-within)
            assert count>0
            parts.append(data[within:within+count]);self.position+=count;remaining-=count
        return b"".join(parts)
