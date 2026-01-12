import { useRef } from "react";
import { useTexture, Float, PerspectiveCamera } from "@react-three/drei";
import * as THREE from "three";

interface ImageCardProps {
  url: string;
}

const ImageCard = ({ url }: ImageCardProps) => {
  const texture = useTexture(url);
  const meshRef = useRef<THREE.Mesh>(null);

  return (
    <Float speed={2} rotationIntensity={0.5}>
      <mesh ref={meshRef} rotation={[0, -0.2, 0]}>
        <planeGeometry args={[3, 3]} />
        <meshBasicMaterial map={texture} side={THREE.DoubleSide} transparent />
      </mesh>
    </Float>
  );
};

export const Scene = ({ imageUrl }: { imageUrl: string | null }) => {
  return (
    <>
      <PerspectiveCamera makeDefault position={[0, 0, 5]} />
      <ambientLight intensity={0.5} />
      {imageUrl && <ImageCard url={imageUrl} />}
    </>
  );
};
