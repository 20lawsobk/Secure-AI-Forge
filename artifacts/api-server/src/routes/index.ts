import { Router, type IRouter } from "express";
import healthRouter from "./health";
import modelProxyRouter from "./model-proxy";
import multimodalRouter from "./multimodal";

const router: IRouter = Router();

router.use(healthRouter);
router.use(multimodalRouter);
router.use(modelProxyRouter);

export default router;
