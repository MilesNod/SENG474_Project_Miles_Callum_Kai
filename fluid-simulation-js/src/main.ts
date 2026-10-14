import { Fluid } from "./fluid";
import { Visualizer } from "./visualizer";

const canvas = document.querySelector("#fluid-canvas") as HTMLCanvasElement;
const ctx = canvas.getContext("2d") as CanvasRenderingContext2D;

const fluid = new Fluid(10, 10);
const visualizer = new Visualizer(fluid);
visualizer.draw_divergence(ctx);
visualizer.draw_grid(ctx);
visualizer.draw_velocities(ctx);