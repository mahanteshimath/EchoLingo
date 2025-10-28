
import React, { useState, useRef, useCallback, useEffect } from 'react';
import { GoogleGenAI, LiveSession, LiveServerMessage, Modality, Blob } from '@google/genai';
import { MicrophoneIcon, StopIcon, SpeakerIcon } from './components/Icons';

// --- Types ---
type TranscriptItem = {
  speaker: 'user' | 'ai' | 'system';
  text: string;
};

// --- Audio Helper Functions ---
function encode(bytes: Uint8Array): string {
  let binary = '';
  const len = bytes.byteLength;
  for (let i = 0; i < len; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

function decode(base64: string): Uint8Array {
  const binaryString = atob(base64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes;
}

async function decodeAudioData(
  data: Uint8Array,
  ctx: AudioContext,
  sampleRate: number,
  numChannels: number,
): Promise<AudioBuffer> {
  const dataInt16 = new Int16Array(data.buffer);
  const frameCount = dataInt16.length / numChannels;
  const buffer = ctx.createBuffer(numChannels, frameCount, sampleRate);

  for (let channel = 0; channel < numChannels; channel++) {
    const channelData = buffer.getChannelData(channel);
    for (let i = 0; i < frameCount; i++) {
      channelData[i] = dataInt16[i * numChannels + channel] / 32768.0;
    }
  }
  return buffer;
}

function createBlob(data: Float32Array): Blob {
    const l = data.length;
    const int16 = new Int16Array(l);
    for (let i = 0; i < l; i++) {
        int16[i] = data[i] < 0 ? data[i] * 32768 : data[i] * 32767;
    }
    return {
        data: encode(new Uint8Array(int16.buffer)),
        mimeType: 'audio/pcm;rate=16000',
    };
}


// --- Language Options ---
const languageOptions = {
    'English': "You are a friendly and patient English teacher. Your goal is to help me practice my English conversational skills. Keep your responses concise and engaging. Ask questions to keep the conversation going.",
    'Spanish': "Eres un profesor de español amable y paciente. Tu objetivo es ayudarme a practicar mis habilidades de conversación en español. Mantén tus respuestas concisas y atractivas. Haz preguntas para mantener la conversación.",
    'French': "Vous êtes un professeur de français sympathique et patient. Votre objectif est de m'aider à pratiquer mes compétences conversationnelles en français. Gardez vos réponses concises et engageantes. Posez des questions pour maintenir la conversation.",
    'German': "Sie sind ein freundlicher und geduldiger Deutschlehrer. Ihr Ziel ist es, mir zu helfen, meine Konversationsfähigkeiten in Deutsch zu üben. Halten Sie Ihre Antworten kurz und ansprechend. Stellen Sie Fragen, um das Gespräch am Laufen zu halten."
};

type Language = keyof typeof languageOptions;

// --- Main App Component ---
export default function App() {
  const [isConnecting, setIsConnecting] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [transcripts, setTranscripts] = useState<TranscriptItem[]>([]);
  const [interimUserTranscript, setInterimUserTranscript] = useState('');
  const [interimAiTranscript, setInterimAiTranscript] = useState('');
  const [selectedLanguage, setSelectedLanguage] = useState<Language>('English');

  const sessionRef = useRef<LiveSession | null>(null);
  const inputAudioContextRef = useRef<AudioContext | null>(null);
  const outputAudioContextRef = useRef<AudioContext | null>(null);
  const microphoneStreamRef = useRef<MediaStream | null>(null);
  const scriptProcessorRef = useRef<ScriptProcessorNode | null>(null);
  const nextStartTimeRef = useRef(0);
  const audioSourcesRef = useRef(new Set<AudioBufferSourceNode>());
  const userInputBufferRef = useRef('');
  const aiOutputBufferRef = useRef('');
  const transcriptEndRef = useRef<HTMLDivElement | null>(null);

  const scrollToBottom = () => {
    transcriptEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(scrollToBottom, [transcripts, interimAiTranscript, interimUserTranscript]);

  const stopConversation = useCallback((err?: Error | string) => {
    setIsConnecting(false);
    setIsConnected(false);
    if (err) {
      const errorMessage = typeof err === 'string' ? err : err.message;
      setError(`Connection failed: ${errorMessage}`);
      setTranscripts(prev => [...prev, { speaker: 'system', text: `Error: ${errorMessage}` }]);
    }

    sessionRef.current?.close();
    sessionRef.current = null;

    microphoneStreamRef.current?.getTracks().forEach(track => track.stop());
    microphoneStreamRef.current = null;
    
    scriptProcessorRef.current?.disconnect();
    scriptProcessorRef.current = null;

    inputAudioContextRef.current?.close().catch(console.error);
    inputAudioContextRef.current = null;
    outputAudioContextRef.current?.close().catch(console.error);
    outputAudioContextRef.current = null;

    audioSourcesRef.current.forEach(source => source.stop());
    audioSourcesRef.current.clear();
    nextStartTimeRef.current = 0;
  }, []);

  const startConversation = useCallback(async () => {
    setError(null);
    setTranscripts([]);
    setInterimUserTranscript('');
    setInterimAiTranscript('');
    userInputBufferRef.current = '';
    aiOutputBufferRef.current = '';
    setIsConnecting(true);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      microphoneStreamRef.current = stream;

      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY as string });

      inputAudioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
      outputAudioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
      
      const sessionPromise = ai.live.connect({
        model: 'gemini-2.5-flash-native-audio-preview-09-2025',
        config: {
          responseModalities: [Modality.AUDIO],
          inputAudioTranscription: {},
          outputAudioTranscription: {},
          speechConfig: {
            voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Zephyr' } },
          },
          systemInstruction: languageOptions[selectedLanguage],
        },
        callbacks: {
          onopen: () => {
            setIsConnecting(false);
            setIsConnected(true);
            setTranscripts([{ speaker: 'system', text: 'Connection established. Start speaking.' }]);
            
            const source = inputAudioContextRef.current!.createMediaStreamSource(stream);
            const scriptProcessor = inputAudioContextRef.current!.createScriptProcessor(4096, 1, 1);
            scriptProcessorRef.current = scriptProcessor;

            scriptProcessor.onaudioprocess = (audioProcessingEvent) => {
              const inputData = audioProcessingEvent.inputBuffer.getChannelData(0);
              const pcmBlob = createBlob(inputData);
              sessionPromise.then((session) => {
                session.sendRealtimeInput({ media: pcmBlob });
              });
            };
            source.connect(scriptProcessor);
            scriptProcessor.connect(inputAudioContextRef.current!.destination);
          },
          onmessage: async (message: LiveServerMessage) => {
            if (message.serverContent?.inputTranscription) {
                const text = message.serverContent.inputTranscription.text;
                userInputBufferRef.current += text;
                setInterimUserTranscript(userInputBufferRef.current);
            }
            if (message.serverContent?.outputTranscription) {
                const text = message.serverContent.outputTranscription.text;
                aiOutputBufferRef.current += text;
                setInterimAiTranscript(aiOutputBufferRef.current);
            }
            if (message.serverContent?.turnComplete) {
                const finalUserInput = userInputBufferRef.current.trim();
                const finalAiOutput = aiOutputBufferRef.current.trim();

                setTranscripts(prev => [
                    ...prev,
                    ...(finalUserInput ? [{ speaker: 'user' as 'user', text: finalUserInput }] : []),
                    ...(finalAiOutput ? [{ speaker: 'ai' as 'ai', text: finalAiOutput }] : [])
                ]);

                userInputBufferRef.current = '';
                aiOutputBufferRef.current = '';
                setInterimUserTranscript('');
                setInterimAiTranscript('');
            }
            if (message.serverContent?.interrupted) {
                audioSourcesRef.current.forEach(source => source.stop());
                audioSourcesRef.current.clear();
                nextStartTimeRef.current = 0;
            }
            
            const base64Audio = message.serverContent?.modelTurn?.parts[0]?.inlineData?.data;
            if (base64Audio) {
              const outputCtx = outputAudioContextRef.current!;
              nextStartTimeRef.current = Math.max(nextStartTimeRef.current, outputCtx.currentTime);
              
              const audioBuffer = await decodeAudioData(decode(base64Audio), outputCtx, 24000, 1);
              const source = outputCtx.createBufferSource();
              source.buffer = audioBuffer;
              source.connect(outputCtx.destination);
              source.addEventListener('ended', () => {
                audioSourcesRef.current.delete(source);
              });
              
              source.start(nextStartTimeRef.current);
              nextStartTimeRef.current += audioBuffer.duration;
              audioSourcesRef.current.add(source);
            }
          },
          onclose: () => {
            stopConversation('Connection closed.');
          },
          onerror: (e: ErrorEvent) => {
            stopConversation(new Error(e.message));
          },
        },
      });

      sessionPromise.then(session => {
        sessionRef.current = session;
      }).catch(err => {
        stopConversation(err);
      });

    } catch (err: any) {
        stopConversation(err);
    }
  }, [selectedLanguage, stopConversation]);

  const handleToggleConversation = useCallback(() => {
    if (isConnected || isConnecting) {
      stopConversation();
    } else {
      startConversation();
    }
  }, [isConnected, isConnecting, startConversation, stopConversation]);

  const getStatus = () => {
      if (error) return { text: "Error", color: "bg-red-500" };
      if (isConnecting) return { text: "Connecting...", color: "bg-yellow-500 animate-pulse" };
      if (isConnected) return { text: "Connected", color: "bg-green-500" };
      return { text: "Ready", color: "bg-gray-500" };
  }

  const status = getStatus();

  return (
    <div className="flex flex-col h-screen bg-gray-900 text-gray-100 p-4 max-w-3xl mx-auto w-full">
      <header className="flex-shrink-0 flex items-center justify-between pb-4 border-b border-gray-700">
        <h1 className="text-2xl font-bold text-sky-400">EchoLingo AI</h1>
        <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <div className={`w-3 h-3 rounded-full ${status.color}`}></div>
              <span className="text-sm text-gray-400">{status.text}</span>
            </div>
            <select
                value={selectedLanguage}
                onChange={(e) => setSelectedLanguage(e.target.value as Language)}
                disabled={isConnected || isConnecting}
                className="bg-gray-800 border border-gray-600 rounded-md px-3 py-1.5 text-sm focus:ring-2 focus:ring-sky-500 focus:outline-none disabled:opacity-50"
            >
                {Object.keys(languageOptions).map(lang => (
                    <option key={lang} value={lang}>{lang}</option>
                ))}
            </select>
        </div>
      </header>

      <main className="flex-grow overflow-y-auto my-4 pr-2 space-y-4">
        {error && <div className="bg-red-900/50 text-red-300 p-3 rounded-lg text-sm">{error}</div>}
        {transcripts.map((item, index) => (
          <div key={index} className={`flex items-start gap-3 ${item.speaker === 'user' ? 'justify-end' : 'justify-start'}`}>
            {item.speaker === 'ai' && <div className="w-8 h-8 rounded-full bg-sky-500 flex-shrink-0 flex items-center justify-center"><SpeakerIcon className="w-5 h-5"/></div>}
            <div className={`max-w-md p-3 rounded-2xl ${
                item.speaker === 'user' ? 'bg-indigo-600 rounded-br-none' :
                item.speaker === 'ai' ? 'bg-gray-700 rounded-bl-none' :
                'bg-gray-800 text-center text-xs text-gray-400 w-full'
              }`}>
              <p className="text-base">{item.text}</p>
            </div>
             {item.speaker === 'user' && <div className="w-8 h-8 rounded-full bg-indigo-600 flex-shrink-0 flex items-center justify-center"><MicrophoneIcon className="w-5 h-5"/></div>}
          </div>
        ))}

        {interimUserTranscript && (
          <div className="flex items-start gap-3 justify-end">
            <div className="max-w-md p-3 rounded-2xl bg-indigo-600/50 rounded-br-none text-gray-400">
                <p className="text-base">{interimUserTranscript}</p>
            </div>
            <div className="w-8 h-8 rounded-full bg-indigo-600/50 flex-shrink-0 flex items-center justify-center"><MicrophoneIcon className="w-5 h-5"/></div>
          </div>
        )}
        {interimAiTranscript && (
          <div className="flex items-start gap-3 justify-start">
            <div className="w-8 h-8 rounded-full bg-sky-500/50 flex-shrink-0 flex items-center justify-center"><SpeakerIcon className="w-5 h-5"/></div>
            <div className="max-w-md p-3 rounded-2xl bg-gray-700/50 rounded-bl-none text-gray-400">
                <p className="text-base">{interimAiTranscript}</p>
            </div>
          </div>
        )}
        <div ref={transcriptEndRef} />
      </main>

      <footer className="flex-shrink-0 flex items-center justify-center pt-4 border-t border-gray-700">
        <button
          onClick={handleToggleConversation}
          disabled={isConnecting}
          className={`w-20 h-20 rounded-full flex items-center justify-center transition-all duration-300 focus:outline-none focus:ring-4
            ${isConnected ? 'bg-red-600 hover:bg-red-700 focus:ring-red-500' : 'bg-sky-600 hover:bg-sky-700 focus:ring-sky-500'}
            ${isConnecting ? 'bg-yellow-500 cursor-not-allowed animate-pulse' : ''}
            ${isConnected ? 'animate-pulse' : ''}
          `}
          aria-label={isConnected ? "Stop conversation" : "Start conversation"}
        >
          {isConnected ? <StopIcon className="w-10 h-10"/> : <MicrophoneIcon className="w-10 h-10"/>}
        </button>
      </footer>
    </div>
  );
}
