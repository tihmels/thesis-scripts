FROM bitnami/minideb as build

RUN apt-get update && apt-get install -y unzip && rm -rf /var/lib/apt/lists/* 

ADD https://github.com/DeutscheKI/tevr-asr-tool/releases/download/v1.0.0/tevr_asr_tool-1.0.0-Linux-x86_64.zip.001 .
ADD https://github.com/DeutscheKI/tevr-asr-tool/releases/download/v1.0.0/tevr_asr_tool-1.0.0-Linux-x86_64.zip.002 .
ADD https://github.com/DeutscheKI/tevr-asr-tool/releases/download/v1.0.0/tevr_asr_tool-1.0.0-Linux-x86_64.zip.003 .
ADD https://github.com/DeutscheKI/tevr-asr-tool/releases/download/v1.0.0/tevr_asr_tool-1.0.0-Linux-x86_64.zip.004 .
ADD https://github.com/DeutscheKI/tevr-asr-tool/releases/download/v1.0.0/tevr_asr_tool-1.0.0-Linux-x86_64.zip.005 .

RUN cat tevr_asr_tool-1.0.0-Linux-x86_64.zip.00* > tevr_asr_tool-1.0.0-Linux-x86_64.zip && unzip tevr_asr_tool-1.0.0-Linux-x86_64.zip

FROM bitnami/minideb

COPY --from=build /tevr_asr_tool-1.0.0-Linux-x86_64.deb /tevr_asr_tool-1.0.0-Linux-x86_64.deb 

RUN dpkg -i tevr_asr_tool-1.0.0-Linux-x86_64.deb && rm tevr_asr_tool-1.0.0-Linux-x86_64.deb

ENTRYPOINT ["tevr_asr_tool"]
CMD ["--help"]