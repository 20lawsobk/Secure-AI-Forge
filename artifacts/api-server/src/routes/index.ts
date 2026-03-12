import { Router, type IRouter } from "express";
import healthRouter from "./health";
import modelProxyRouter from "./model-proxy";

const router: IRouter = Router();

router.use(healthRouter);
router.use(modelProxyRouter);

export default router;
